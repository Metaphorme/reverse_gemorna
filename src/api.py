"""FastAPI application for GEMORNA generation and prediction workflows."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .gemorna_services import GemornaService
except ImportError:  # Supports running tests with src/ on sys.path.
    from gemorna_services import GemornaService


app = FastAPI(
    title="GEMORNA REST API",
    version="0.1.0",
    description="REST API for GEMORNA CDS and UTR generation and scoring.",
)

_service = GemornaService()


class CDSGenerationRequest(BaseModel):
    protein_sequence: str = Field(..., min_length=1)
    seed: Optional[int] = None


class CDSGenerationResponse(BaseModel):
    implementation: Literal["open", "closed"]
    protein_sequence: str
    dna_sequence: str
    rna_sequence: str
    naturalness: float
    sampling_seed: int
    device: str


class UTRGenerationRequest(BaseModel):
    length: Literal["short", "medium", "long"]
    seed: Optional[int] = None


class UTRGenerationResponse(BaseModel):
    utr_type: Literal["5utr", "3utr"]
    length: Literal["short", "medium", "long"]
    sequence: str
    score: float
    sampling_seed: Optional[int] = None
    device: str


class UTRScoreRequest(BaseModel):
    sequence: str = Field(..., min_length=1)


class UTRScoreResponse(BaseModel):
    utr_type: Literal["5utr", "3utr"]
    sequence: str
    score: float
    device: str


def get_service() -> GemornaService:
    return _service


def _http_400_on_value_error(callback):
    try:
        return callback()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _cds_response(result) -> CDSGenerationResponse:
    return CDSGenerationResponse(
        implementation=result.implementation,
        protein_sequence=result.protein_sequence,
        dna_sequence=result.dna_sequence,
        rna_sequence=result.rna_sequence,
        naturalness=result.naturalness,
        sampling_seed=result.sampling_seed,
        device=result.device,
    )


def _utr_generation_response(result) -> UTRGenerationResponse:
    return UTRGenerationResponse(
        utr_type=result.utr_type,
        length=result.length,
        sequence=result.sequence,
        score=result.score,
        sampling_seed=result.sampling_seed,
        device=result.device,
    )


def _utr_score_response(result) -> UTRScoreResponse:
    return UTRScoreResponse(
        utr_type=result.utr_type,
        sequence=result.sequence,
        score=result.score,
        device=result.device,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/cds/open/generate", response_model=CDSGenerationResponse)
def generate_cds_open(
    request: CDSGenerationRequest,
    service: GemornaService = Depends(get_service),
):
    return _http_400_on_value_error(
        lambda: _cds_response(
            service.generate_cds_open(request.protein_sequence, seed=request.seed)
        )
    )


@app.post("/api/v1/cds/closed/generate", response_model=CDSGenerationResponse)
def generate_cds_closed(
    request: CDSGenerationRequest,
    service: GemornaService = Depends(get_service),
):
    return _http_400_on_value_error(
        lambda: _cds_response(
            service.generate_cds_closed(request.protein_sequence, seed=request.seed)
        )
    )


@app.post("/api/v1/utr/5/generate", response_model=UTRGenerationResponse)
def generate_5utr(
    request: UTRGenerationRequest,
    service: GemornaService = Depends(get_service),
):
    return _http_400_on_value_error(
        lambda: _utr_generation_response(
            service.generate_utr("5utr", request.length, seed=request.seed)
        )
    )


@app.post("/api/v1/utr/3/generate", response_model=UTRGenerationResponse)
def generate_3utr(
    request: UTRGenerationRequest,
    service: GemornaService = Depends(get_service),
):
    return _http_400_on_value_error(
        lambda: _utr_generation_response(
            service.generate_utr("3utr", request.length, seed=request.seed)
        )
    )


@app.post("/api/v1/utr/5/score", response_model=UTRScoreResponse)
def score_5utr(
    request: UTRScoreRequest,
    service: GemornaService = Depends(get_service),
):
    return _http_400_on_value_error(
        lambda: _utr_score_response(service.score_utr("5utr", request.sequence))
    )


@app.post("/api/v1/utr/3/score", response_model=UTRScoreResponse)
def score_3utr(
    request: UTRScoreRequest,
    service: GemornaService = Depends(get_service),
):
    return _http_400_on_value_error(
        lambda: _utr_score_response(service.score_utr("3utr", request.sequence))
    )
