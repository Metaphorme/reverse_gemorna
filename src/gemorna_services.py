"""Reusable service layer for GEMORNA generation and prediction workflows."""

from __future__ import annotations

import math
import pickle
import random
import sys
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CDS_CKPT = REPO_ROOT / "checkpoints" / "gemorna_cds.pt"
DEFAULT_5UTR_GEN_CKPT = REPO_ROOT / "checkpoints" / "gemorna_5utr.pt"
DEFAULT_3UTR_GEN_CKPT = REPO_ROOT / "checkpoints" / "gemorna_3utr.pt"
DEFAULT_5UTR_PRED_CKPT = REPO_ROOT / "checkpoints" / "5utr.pt"
DEFAULT_3UTR_PRED_CKPT = REPO_ROOT / "checkpoints" / "3utr.pt"

VALID_UTR_LENGTHS = {"short", "medium", "long"}
VALID_UTR_TYPES = {"5utr", "3utr"}


@dataclass(frozen=True)
class CDSServiceResult:
    implementation: str
    protein_sequence: str
    dna_sequence: str
    rna_sequence: str
    naturalness: float
    sampling_seed: int
    device: str


@dataclass(frozen=True)
class UTRGenerationResult:
    utr_type: str
    length: str
    sequence: str
    score: float
    sampling_seed: int | None
    device: str


@dataclass(frozen=True)
class UTRGeneratedSequence:
    utr_type: str
    length: str
    sequence: str
    sampling_seed: int | None
    device: str


@dataclass(frozen=True)
class UTRScoreResult:
    utr_type: str
    sequence: str
    score: float
    device: str


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def validate_utr_length(length: str) -> str:
    if length not in VALID_UTR_LENGTHS:
        raise ValueError("UTR length must be short, medium, or long.")
    return length


def validate_utr_type(utr_type: str) -> str:
    if utr_type not in VALID_UTR_TYPES:
        raise ValueError("UTR type must be 5utr or 3utr.")
    return utr_type


def normalize_utr_sequence(sequence: str) -> str:
    if not sequence:
        raise ValueError("No input UTR sequence.")

    normalized = sequence.upper().replace("T", "U")
    invalid = sorted(set(normalized) - set("ACGUN"))
    if invalid:
        invalid_text = ", ".join(invalid)
        raise ValueError(f"UTR sequence contains invalid characters: {invalid_text}.")
    return normalized


def parse_generated_utr_output(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) < 2 or lines[-2] != "Generated UTR:":
        raise ValueError("Unable to parse generated UTR sequence from model output.")
    return lines[-1].replace("T", "U").upper()


def _sampling_seed(seed: int | None) -> int:
    if seed is None:
        return random.randint(1, 200)

    state = random.getstate()
    try:
        random.seed(seed)
        return random.randint(1, 200)
    finally:
        random.setstate(state)


@contextlib.contextmanager
def _temporary_random_seed(seed: int | None):
    if seed is None:
        yield
        return

    state = random.getstate()
    try:
        random.seed(seed)
        yield
    finally:
        random.setstate(state)


def _stoi(vocab, token):
    if hasattr(vocab, "stoi"):
        return vocab.stoi[token]
    return vocab[token]


def _load_cds_vocabularies():
    with open(REPO_ROOT / "vocab" / "prot_vocab.pkl", "rb") as f:
        prot_vocab = pickle.load(f)
    with open(REPO_ROOT / "vocab" / "cds_vocab.pkl", "rb") as f:
        cds_vocab = pickle.load(f)
    return prot_vocab, cds_vocab


def build_closed_cds_model(ckpt_path=DEFAULT_CDS_CKPT, device=None):
    import torch
    from config import GEMORNA_CDS_Config
    from models.gemorna_cds import Decoder, Encoder
    from shared.mod_xzr01 import CDS

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

    model = CDS(enc, dec, cfg.prot_pad_idx, cfg.cds_pad_idx, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _utr_config_vocab_and_ckpt(utr_type: str):
    from config import (
        GEMORNA_3UTR_Config,
        GEMORNA_5UTR_Config,
        five_prime_utr_vocab,
        three_prime_utr_vocab,
    )

    utr_type = validate_utr_type(utr_type)
    if utr_type == "5utr":
        return GEMORNA_5UTR_Config(), five_prime_utr_vocab, DEFAULT_5UTR_GEN_CKPT
    return GEMORNA_3UTR_Config(), three_prime_utr_vocab, DEFAULT_3UTR_GEN_CKPT


def build_closed_utr_model(utr_type: str, ckpt_path=None, device=None):
    import torch
    from shared.mod_xzr01 import UTR

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, _, default_ckpt = _utr_config_vocab_and_ckpt(utr_type)
    ckpt_path = resolve_path(ckpt_path or default_ckpt)

    model = UTR(config)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model


class ClosedCDSGenerator:
    """Structured wrapper around the closed CDS Cython implementation."""

    def __init__(self, ckpt_path=DEFAULT_CDS_CKPT, device=None):
        import torch

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prot_vocab, self.cds_vocab = _load_cds_vocabularies()
        self.model = build_closed_cds_model(ckpt_path=ckpt_path, device=self.device)

    def translate_protein_to_rna(self, protein_sequence: str, seed: int | None = None):
        import torch
        from config import eos_token, init_token
        from open_cds_generator import has_noncanonical
        from tokenization import tokenize_aa
        from utils.utils_cds import trunc_protein_seq

        if not protein_sequence:
            raise ValueError("Please provide the protein sequence.")
        if has_noncanonical(protein_sequence):
            raise ValueError(
                "The input protein sequence contains non-canonical amino acid characters."
            )

        self.model.eval()
        sampling_seed = _sampling_seed(seed)
        generated_seqs = []
        final_modelscore = 0.0

        with torch.no_grad():
            for seq in trunc_protein_seq(protein_sequence):
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

        if not generated_seqs:
            raise ValueError("Protein sequence did not produce any CDS codons.")

        dna_sequence = "".join(generated_seqs).upper()
        naturalness = math.exp(final_modelscore / len(generated_seqs))
        return CDSServiceResult(
            implementation="closed",
            protein_sequence=protein_sequence,
            dna_sequence=dna_sequence,
            rna_sequence=dna_sequence.replace("T", "U"),
            naturalness=naturalness,
            sampling_seed=sampling_seed,
            device=str(self.device),
        )


class ClosedUTRGenerator:
    """Structured wrapper around the closed UTR generation implementation."""

    def __init__(self, device=None):
        import torch

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models = {}

    def _get_model_and_vocab(self, utr_type: str):
        config, vocab, ckpt_path = _utr_config_vocab_and_ckpt(utr_type)
        del config
        if utr_type not in self._models:
            self._models[utr_type] = build_closed_utr_model(
                utr_type,
                ckpt_path=ckpt_path,
                device=self.device,
            )
        return self._models[utr_type], vocab

    def generate(self, utr_type: str, length: str, seed: int | None = None):
        import torch

        utr_type = validate_utr_type(utr_type)
        length = validate_utr_length(length)
        model, vocab = self._get_model_and_vocab(utr_type)
        sampling_seed = _sampling_seed(seed) if seed is not None else None

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), torch.no_grad(), _temporary_random_seed(seed):
            model.gen(utr_type, vocab, self.device, length)

        sequence = parse_generated_utr_output(buffer.getvalue())
        return UTRGeneratedSequence(
            utr_type=utr_type,
            length=length,
            sequence=sequence,
            sampling_seed=sampling_seed,
            device=str(self.device),
        )


class UTRScorer:
    def __init__(self, utr_type: str, ckpt_path=None, device=None):
        import torch

        self.utr_type = validate_utr_type(utr_type)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = resolve_path(
            ckpt_path
            or (DEFAULT_5UTR_PRED_CKPT if self.utr_type == "5utr" else DEFAULT_3UTR_PRED_CKPT)
        )
        self.model = self._build_model()

    def _build_model(self):
        import torch
        import shared.helper as helper

        if self.utr_type == "5utr":
            import models.model_pred5UTR as model_module

            args = SimpleNamespace(
                embed_num=10,
                embed_dim=64,
                kernel_num=128,
                kernel_sizes=helper.kernel_sizes_5UTR,
                dropout=0.1,
            )
        else:
            import models.model_pred3UTR as model_module

            args = SimpleNamespace(
                embed_num=10,
                embed_dim=256,
                kernel_num=200,
                kernel_sizes=helper.kernel_sizes_3UTR,
                dropout=0.1,
            )

        predictor = model_module.Model(args).to(self.device)
        predictor.load_state_dict(torch.load(self.ckpt_path, map_location=self.device), strict=True)
        predictor.eval()
        return predictor

    def score_sequence(self, sequence: str):
        import torch
        import shared.helper as helper

        normalized = normalize_utr_sequence(sequence)
        tokenized_seq = helper.tokenize(normalized)
        if self.utr_type == "5utr":
            padded = tokenized_seq + [helper.vocab["[PAD]"]] * max(0, 100 - len(tokenized_seq))
            model_input = torch.tensor([padded], device=self.device)
        else:
            model_input = torch.tensor([tokenized_seq], device=self.device)

        with torch.no_grad():
            pred = self.model(model_input).squeeze().cpu().numpy()

        if self.utr_type == "5utr":
            score = helper.scale(pred)
        else:
            score = pred

        return UTRScoreResult(
            utr_type=self.utr_type,
            sequence=normalized,
            score=float(score),
            device=str(self.device),
        )


class GemornaService:
    def __init__(
        self,
        *,
        device=None,
        open_cds_generator=None,
        closed_cds_generator=None,
        utr_generator=None,
        utr_scorers=None,
    ):
        self.device = device
        self._open_cds_generator = open_cds_generator
        self._closed_cds_generator = closed_cds_generator
        self._utr_generator = utr_generator
        self._utr_scorers = dict(utr_scorers or {})

    def _get_open_cds_generator(self):
        if self._open_cds_generator is None:
            from open_cds_generator import OpenCDSGenerator

            self._open_cds_generator = OpenCDSGenerator(
                ckpt_path=DEFAULT_CDS_CKPT,
                device=self.device,
            )
        return self._open_cds_generator

    def _get_closed_cds_generator(self):
        if self._closed_cds_generator is None:
            self._closed_cds_generator = ClosedCDSGenerator(
                ckpt_path=DEFAULT_CDS_CKPT,
                device=self.device,
            )
        return self._closed_cds_generator

    def _get_utr_generator(self):
        if self._utr_generator is None:
            self._utr_generator = ClosedUTRGenerator(device=self.device)
        return self._utr_generator

    def _get_utr_scorer(self, utr_type: str):
        utr_type = validate_utr_type(utr_type)
        if utr_type not in self._utr_scorers:
            self._utr_scorers[utr_type] = UTRScorer(utr_type, device=self.device)
        return self._utr_scorers[utr_type]

    @staticmethod
    def _cds_result(implementation: str, result) -> CDSServiceResult:
        return CDSServiceResult(
            implementation=implementation,
            protein_sequence=result.protein_sequence,
            dna_sequence=result.dna_sequence,
            rna_sequence=result.rna_sequence,
            naturalness=float(result.naturalness),
            sampling_seed=int(result.sampling_seed),
            device=str(result.device),
        )

    def generate_cds_open(self, protein_sequence: str, seed: int | None = None):
        result = self._get_open_cds_generator().translate_protein_to_rna(
            protein_sequence,
            seed=seed,
        )
        return self._cds_result("open", result)

    def generate_cds_closed(self, protein_sequence: str, seed: int | None = None):
        result = self._get_closed_cds_generator().translate_protein_to_rna(
            protein_sequence,
            seed=seed,
        )
        return self._cds_result("closed", result)

    def score_utr(self, utr_type: str, sequence: str):
        utr_type = validate_utr_type(utr_type)
        result = self._get_utr_scorer(utr_type).score_sequence(sequence)
        return UTRScoreResult(
            utr_type=utr_type,
            sequence=result.sequence,
            score=float(result.score),
            device=str(result.device),
        )

    def generate_utr(self, utr_type: str, length: str, seed: int | None = None):
        generated = self._get_utr_generator().generate(
            validate_utr_type(utr_type),
            validate_utr_length(length),
            seed=seed,
        )
        scored = self.score_utr(utr_type, generated.sequence)
        return UTRGenerationResult(
            utr_type=utr_type,
            length=generated.length,
            sequence=generated.sequence,
            score=scored.score,
            sampling_seed=generated.sampling_seed,
            device=str(generated.device),
        )
