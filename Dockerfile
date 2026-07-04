FROM continuumio/miniconda3:24.5.0-0

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yaml ./
RUN conda env create -f environment.yaml \
    && conda clean -afy

COPY src ./src
COPY checkpoints ./checkpoints
COPY vocab ./vocab
COPY README.md README_OLD.md LICENSE ./
COPY docs/rest-api.md ./docs/rest-api.md

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["conda", "run", "--no-capture-output", "-n", "gemorna", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
