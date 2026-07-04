#!/usr/bin/env python3
"""Open CDS generator interface for GEMORNA.

This module intentionally avoids importing ``shared.mod_xzr01``.  It exposes a
small Python API for translating an amino-acid sequence into an RNA CDS
sequence and a CLI similar to ``src/generate.py``.
"""

import argparse
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from config import GEMORNA_CDS_Config, eos_token, init_token
from models.gemorna_cds import Decoder, Encoder
from tokenization import tokenize_aa
from utils.utils_cds import CDS_ as CDSOpen
from utils.utils_cds import trunc_protein_seq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT_PATH = REPO_ROOT / "checkpoints" / "gemorna_cds.pt"


@dataclass(frozen=True)
class CDSGenerationResult:
    """Generated CDS payload returned by the open API."""

    protein_sequence: str
    dna_sequence: str
    rna_sequence: str
    naturalness: float
    sampling_seed: int
    device: str


def has_noncanonical(protein_seq):
    canonical = set("ACDEFGHIKLMNPQRSTVWY*")
    return any(residue not in canonical for residue in protein_seq.upper())


def resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def _stoi(vocab, token):
    if hasattr(vocab, "stoi"):
        return vocab.stoi[token]
    return vocab[token]


def _load_vocabularies():
    with open(REPO_ROOT / "vocab" / "prot_vocab.pkl", "rb") as f:
        prot_vocab = pickle.load(f)
    with open(REPO_ROOT / "vocab" / "cds_vocab.pkl", "rb") as f:
        cds_vocab = pickle.load(f)
    return prot_vocab, cds_vocab


def build_open_cds_model(ckpt_path=DEFAULT_CKPT_PATH, device=None):
    """Build and load the open CDS model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_path(ckpt_path)
    cfg = GEMORNA_CDS_Config()

    enc = Encoder(
        input_dim=cfg.input_dim,
        hid_dim=cfg.hidden_dim,
        n_layers=cfg.num_layers,
        n_heads=cfg.num_heads,
        pf_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        cnn_kernel_size=cfg.cnn_kernel_size,
        cnn_padding=cfg.cnn_padding,
        device=device,
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

    model = CDSOpen(enc, dec, cfg.prot_pad_idx, cfg.cds_pad_idx, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _sampling_seed(seed):
    if seed is None:
        return random.randint(1, 200)

    state = random.getstate()
    try:
        random.seed(seed)
        return random.randint(1, 200)
    finally:
        random.setstate(state)


class OpenCDSGenerator:
    """Reusable open CDS generator loaded from a checkpoint."""

    def __init__(self, ckpt_path=DEFAULT_CKPT_PATH, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prot_vocab, self.cds_vocab = _load_vocabularies()
        self.model = build_open_cds_model(ckpt_path=ckpt_path, device=self.device)

    @torch.no_grad()
    def translate_protein_to_rna(self, protein_seq, seed=None):
        """Translate an amino-acid sequence into RNA CDS and naturalness score."""
        if protein_seq is None:
            raise ValueError("Please provide the protein sequence.")
        if has_noncanonical(protein_seq):
            raise ValueError(
                "The input protein sequence contains non-canonical amino acid characters."
            )

        self.model.eval()
        sampling_seed = _sampling_seed(seed)
        generated_seqs = []
        final_modelscore = 0.0

        for seq in trunc_protein_seq(protein_seq):
            protein_tokens = tokenize_aa(seq)
            tokens = [init_token] + protein_tokens + [eos_token]
            prot_indexes = [_stoi(self.prot_vocab, token) for token in tokens]

            prot_tensor = torch.LongTensor(prot_indexes).unsqueeze(0).to(self.device)
            prot_mask = self.model.make_prot_mask(prot_tensor)
            enc_prot = self.model.encoder(prot_tensor, prot_mask)
            generated_seq, model_score = self.model.sampling(
                enc_prot, prot_mask, tokens, self.cds_vocab, self.device, sampling_seed
            )
            generated_seqs.extend(generated_seq)
            final_modelscore += model_score

        dna_sequence = "".join(generated_seqs).upper()
        naturalness = math.exp(final_modelscore / len(generated_seqs))
        return CDSGenerationResult(
            protein_sequence=protein_seq,
            dna_sequence=dna_sequence,
            rna_sequence=dna_sequence.replace("T", "U"),
            naturalness=naturalness,
            sampling_seed=sampling_seed,
            device=str(self.device),
        )


def translate_protein_to_rna(
    protein_seq,
    ckpt_path=DEFAULT_CKPT_PATH,
    device=None,
    seed=None,
):
    """Convenience function for one-shot amino-acid to RNA CDS generation."""
    generator = OpenCDSGenerator(ckpt_path=ckpt_path, device=device)
    return generator.translate_protein_to_rna(protein_seq, seed=seed)


def main(args):
    if args.mode != "cds":
        raise ValueError("open_cds_generator.py only supports --mode cds")

    result = translate_protein_to_rna(
        args.protein_seq,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
    )

    if args.output == "dna":
        sequence = result.dna_sequence
        label = "Generated CDS DNA & Naturalness"
    elif args.output == "both":
        print("\nGenerated CDS DNA & Naturalness")
        print(f"{result.dna_sequence} {result.naturalness:.2f}")
        print("\nGenerated CDS RNA & Naturalness")
        print(f"{result.rna_sequence} {result.naturalness:.2f}\n")
        return result
    else:
        sequence = result.rna_sequence
        label = "Generated CDS RNA & Naturalness"

    print(f"\n{label}")
    print(f"{sequence} {result.naturalness:.2f}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GEMORNA open CDS generator")
    parser.add_argument(
        "--mode",
        type=str,
        default="cds",
        help="Generation mode. Only 'cds' is supported by this open generator.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(DEFAULT_CKPT_PATH),
        help="Path to the GEMORNA CDS checkpoint.",
    )
    parser.add_argument(
        "--protein_seq",
        type=str,
        required=True,
        help="Input amino-acid sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional Python random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        choices=("rna", "dna", "both"),
        default="rna",
        help="Sequence alphabet to print.",
    )
    main(parser.parse_args())
