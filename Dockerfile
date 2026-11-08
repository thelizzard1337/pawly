FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM python:3.12-slim-bookworm AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN groupadd --gid 10001 pawly && useradd --uid 10001 --gid 10001 --no-create-home pawly

# Pinned official torchvision weights, verified by full SHA-256 (standard library only).
# Built once; runtime containers never download weights.
FROM python:3.12-slim-bookworm AS weights
WORKDIR /build
COPY app/cv/weights.py ./weights.py
RUN python weights.py download --dir /opt/pawly/weights efficientnet_b0 resnet50 ssdlite \
    && chmod 0755 /opt/pawly/weights && chmod 0644 /opt/pawly/weights/*

# Light API/migrate image: no torch, no numpy, no model weights.
FROM base AS runtime
RUN uv sync --frozen --no-dev --no-install-project \
    && mkdir -p /data/media && chown 10001:10001 /data/media
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# CPU torch/torchvision for the CV worker and the training/evaluation CLI.
FROM base AS cv-deps
RUN uv sync --frozen --no-dev --group ml --no-install-project

FROM cv-deps AS cv-runtime
COPY --from=weights /opt/pawly/weights /opt/pawly/weights
RUN mkdir -p /data/media /data/ml && chown 10001:10001 /data/media /data/ml
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
ENV CV_WEIGHTS_DIR=/opt/pawly/weights
USER 10001:10001
CMD ["python", "-m", "app.worker"]

FROM cv-deps AS test
RUN uv sync --frozen --group ml --no-install-project
COPY --from=weights /opt/pawly/weights /opt/pawly/weights
ENV CV_WEIGHTS_DIR=/opt/pawly/weights
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
COPY tests ./tests
COPY scripts ./scripts
RUN mkdir -p /data/media && chmod 777 /data/media
CMD ["python", "-m", "scripts.test_runner"]

# Optional GPU variant for training only (compose.gpu.yaml). Not part of the default build and
# not lock-reproducible: CUDA wheels come from the PyTorch cu130 index at build time.
FROM cv-runtime AS ml-gpu
USER root
RUN uv pip install --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1" "torchvision==0.29.1"
USER 10001:10001
