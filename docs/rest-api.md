# GEMORNA REST API

This API exposes the generation and prediction workflows from `README_OLD.md`.
Run commands from the repository root so checkpoint, vocabulary, and shared
library paths resolve correctly.

## Environment

Create or update the reproducible conda environment:

```bash
conda env create -f environment.yaml
```

For an existing environment:

```bash
conda env update -n gemorna -f environment.yaml
```

## Start The Server

```bash
conda run -n gemorna uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Docker

Build the API image from the repository root:

```bash
docker build -t gemorna-api .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 gemorna-api
```

For NVIDIA GPU access, run with the NVIDIA container runtime on a host with
compatible drivers:

```bash
docker run --rm --gpus all -p 8000:8000 gemorna-api
```

## CDS Generation

Open CDS implementation:

```bash
curl -X POST http://localhost:8000/api/v1/cds/open/generate \
  -H 'Content-Type: application/json' \
  -d '{"protein_sequence":"MVSKGEELFTGVVPILVE","seed":0}'
```

Closed CDS implementation:

```bash
curl -X POST http://localhost:8000/api/v1/cds/closed/generate \
  -H 'Content-Type: application/json' \
  -d '{"protein_sequence":"MVSKGEELFTGVVPILVE","seed":0}'
```

Both endpoints return `dna_sequence`, `rna_sequence`, `naturalness`,
`sampling_seed`, and `device`.

## UTR Generation And Scoring

Generate and score a 5UTR:

```bash
curl -X POST http://localhost:8000/api/v1/utr/5/generate \
  -H 'Content-Type: application/json' \
  -d '{"length":"short","seed":0}'
```

Generate and score a 3UTR:

```bash
curl -X POST http://localhost:8000/api/v1/utr/3/generate \
  -H 'Content-Type: application/json' \
  -d '{"length":"long","seed":0}'
```

`length` must be `short`, `medium`, or `long`. The generated sequence is scored
with the matching predictor before the response is returned.

## UTR Scoring

Score a 5UTR:

```bash
curl -X POST http://localhost:8000/api/v1/utr/5/score \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"TACGTTTTGACCTTCGTTCATTTTG"}'
```

Score a 3UTR:

```bash
curl -X POST http://localhost:8000/api/v1/utr/3/score \
  -H 'Content-Type: application/json' \
  -d '{"sequence":"TGTCCCCGGGTCTTCCAACGGACTGGCGTTGCCCCGGTTCACTGGGGACTGCCCTTGGGGTCTCGCTCACCTTCAGCACACATTATCGGGAGCAGTGTCTTCCATAATGT"}'
```
