#!/usr/bin/env python3
"""Compare open CDS generation against the closed Cython implementation."""

import argparse
import contextlib
import io
import pickle
import random
from pathlib import Path

import torch

from config import GEMORNA_CDS_Config
from models.gemorna_cds import Decoder, Encoder
from open_cds_generator import OpenCDSGenerator, translate_protein_to_rna
from shared.mod_xzr01 import CDS as CDSClosed


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_closed_model(ckpt, device):
    cfg = GEMORNA_CDS_Config()
    enc = Encoder(
        input_dim=cfg.input_dim,
        hid_dim=cfg.hidden_dim,
        n_layers=cfg.num_layers,
        n_heads=cfg.num_heads,
        pf_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        device=device,
        cnn_kernel_size=cfg.cnn_kernel_size,
        cnn_padding=cfg.cnn_padding,
    )
    dec = Decoder(
        output_dim=cfg.output_dim,
        hid_dim=cfg.hidden_dim,
        n_layers=cfg.num_layers,
        n_heads=cfg.num_heads,
        pf_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        device=device,
    )
    model = CDSClosed(enc, dec, cfg.prot_pad_idx, cfg.cds_pad_idx, device)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def load_vocabularies():
    with open(REPO_ROOT / "vocab" / "prot_vocab.pkl", "rb") as f:
        prot_vocab = pickle.load(f)
    with open(REPO_ROOT / "vocab" / "cds_vocab.pkl", "rb") as f:
        cds_vocab = pickle.load(f)
    return prot_vocab, cds_vocab


def capture_closed_gen(model, protein_seq, prot_vocab, cds_vocab, device, seed):
    random.seed(seed)
    torch.manual_seed(12345)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), torch.no_grad():
        model.gen(protein_seq, prot_vocab, cds_vocab, device)
    return buffer.getvalue()


def parse_closed_output(output):
    payload = output.strip().splitlines()[-1]
    seq, naturalness = payload.rsplit(" ", 1)
    return seq, naturalness


def make_logits_hook(target):
    def hook(module, inputs, output):
        target.append(output.detach().cpu().clone())

    return hook


def compare_logits(closed_logits, open_logits):
    if len(closed_logits) != len(open_logits):
        return False, f"call count {len(closed_logits)} != {len(open_logits)}"

    max_diff = 0.0
    for step, (closed, open_) in enumerate(zip(closed_logits, open_logits)):
        if closed.shape != open_.shape:
            return False, f"step {step} shape {tuple(closed.shape)} != {tuple(open_.shape)}"
        diff = (closed - open_).abs().max().item()
        max_diff = max(max_diff, diff)
        if not torch.allclose(closed, open_, atol=1e-6, rtol=1e-5):
            return False, f"step {step} logits differ; max_diff={diff:.8f}"

    return True, f"max_diff={max_diff:.8f}"


def default_cases():
    amino = "ACDEFGHIKLMNPQRSTVWY*"
    rng = random.Random(20260704)
    return [
        ("single_residue", "M"),
        ("stop_residue", "*"),
        ("paper_sample", "MVSKGEELFTGVVPILVE"),
        ("all_canonical", "ACDEFGHIKLMNPQRSTVWY*"),
        ("random_50", "".join(rng.choice(amino) for _ in range(50))),
        ("chunk_boundary", "M" * 170),
        ("one_past_chunk", "M" * 171),
        ("max_position_no_error", "M" * 185),
        ("past_position_limit", "M" * 186),
    ]


def parse_seeds(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser("Compare open and closed CDS generation")
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "gemorna_cds.pt",
    )
    parser.add_argument(
        "--protein_seq",
        type=str,
        help="Optional additional protein sequence to compare",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,199",
        help="Comma-separated Python random seeds used before generation",
    )
    parser.add_argument(
        "--skip_logits",
        action="store_true",
        help="Only compare final sequence and naturalness text",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt_path
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path

    prot_vocab, cds_vocab = load_vocabularies()
    ckpt = torch.load(ckpt_path, map_location=device)
    closed_model = build_closed_model(ckpt, device)
    open_generator = OpenCDSGenerator(ckpt_path=ckpt_path, device=device)

    cases = default_cases()
    if args.protein_seq:
        cases.append(("user_sequence", args.protein_seq))

    failures = []
    total = 0
    seeds = parse_seeds(args.seeds)

    for name, protein_seq in cases:
        for seed in seeds:
            total += 1
            closed_logits = []
            open_logits = []
            closed_hook = None
            open_hook = None
            if not args.skip_logits:
                closed_hook = closed_model.decoder.fc_out.register_forward_hook(
                    make_logits_hook(closed_logits)
                )
                open_hook = open_generator.model.decoder.fc_out.register_forward_hook(
                    make_logits_hook(open_logits)
                )

            try:
                closed_output = capture_closed_gen(
                    closed_model, protein_seq, prot_vocab, cds_vocab, device, seed
                )
                closed_dna, closed_naturalness = parse_closed_output(closed_output)
                open_result = open_generator.translate_protein_to_rna(
                    protein_seq, seed=seed
                )
                function_result = translate_protein_to_rna(
                    protein_seq, ckpt_path=ckpt_path, device=device, seed=seed
                )
            finally:
                if closed_hook is not None:
                    closed_hook.remove()
                if open_hook is not None:
                    open_hook.remove()

            open_naturalness = f"{open_result.naturalness:.2f}"
            if closed_dna != open_result.dna_sequence:
                failures.append(
                    (name, seed, f"dna differs: closed={closed_dna} open={open_result.dna_sequence}")
                )
            if closed_naturalness != open_naturalness:
                failures.append(
                    (
                        name,
                        seed,
                        f"naturalness differs: closed={closed_naturalness} open={open_naturalness}",
                    )
                )
            if open_result.rna_sequence != open_result.dna_sequence.replace("T", "U"):
                failures.append((name, seed, "RNA sequence is not DNA sequence with T->U"))
            if function_result != open_result:
                failures.append((name, seed, "function API result differs from generator API"))

            if not args.skip_logits:
                logits_match, message = compare_logits(closed_logits, open_logits)
                if not logits_match:
                    failures.append((name, seed, message))

    if failures:
        print(f"FAIL: {len(failures)} issue(s) across {total} comparisons")
        for name, seed, message in failures[:12]:
            print(f"case={name} seed={seed}: {message}")
        raise SystemExit(1)

    print(f"PASS: {total} open/closed CDS generation comparisons matched")
    print(f"device={device}")


if __name__ == "__main__":
    main()
