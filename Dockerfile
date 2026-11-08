FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM python:3.12-slim-bookworm AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./

FROM base AS runtime
RUN uv sync --frozen --no-dev --no-install-project \
    && groupadd --gid 10001 pawly \
    && useradd --uid 10001 --gid 10001 --no-create-home pawly \
    && mkdir -p /data/media && chown 10001:10001 /data/media
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS test
RUN uv sync --frozen --no-install-project
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./
COPY tests ./tests
COPY scripts ./scripts
RUN mkdir -p /data/media && chmod 777 /data/media
CMD ["python", "-m", "scripts.test_runner"]
