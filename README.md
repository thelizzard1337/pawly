# PawLy — backend MVP

Backend API for lost/found animal listings and shelter adoption.
The API includes accounts, private contacts, photos, geographic filters, map points,
shelter verification, adoption applications, contact requests, and asynchronous dog photo search.
There is no frontend.

**Photo search uses a real pretrained CV pipeline by default** (`CV_MODE=pretrained_features`):
the user's crop passes a dog gate (SSDLite320, COCO), gets an EfficientNet-B0 (ImageNet) embedding
and an RGB histogram, and active listings are ranked by exact cosine similarity in
PostgreSQL/pgvector. Results are candidates in review order — **no similarity or confidence
values are returned**, and breed/colour similarity does not prove that the same dog was found.
Trained breed/colour heads are an explicit, separately activated experiment (`trained_heads`);
`mock` remains only for plumbing tests and is always labelled `provider="mock-v1"`.

- [docs/ml.md](./docs/ml.md): CV/ML decisions, formal problem statement, modes, dataset, training,
  evaluation, measured results and limitations.
- [docs.md](./docs.md): canonical technical reference and architectural decision log
  (configuration, authentication, data flows, maintenance, backup/restore, conventions).

## Start

Docker Engine and Docker Compose v2 are sufficient. Run from the repository root:

```sh
docker compose up
```

Compose builds the images if they are absent, waits for PostgreSQL/PostGIS/pgvector, applies
Alembic migrations, then starts the API and the CV worker. No host Python, uv, mandatory `.env`,
manual migration, extension setup or preparation script is required.
**The first build requires network access**: it downloads Python packages (CPU-only
torch/torchvision for the worker image) and the pinned torchvision weights (verified by SHA-256
and baked read-only into the worker image). Later starts do not download weights.
No dataset is downloaded and no training runs on `up`.

- API: http://localhost:8000/api/v1 (dog photo search: `/api/v2/photo-searches`)
- Swagger: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/openapi.json
- Liveness: http://localhost:8000/health/live
- Readiness (CRUD): http://localhost:8000/health/ready
- ML readiness: http://localhost:8000/health/ml, index status: `/api/v1/cv/status`

For detached operation and logs:

```sh
docker compose up -d --wait
docker compose logs -f api worker migrate
docker compose ps -a
```

Only the API port is published, on loopback. PostgreSQL is not exposed on the host.
Optional overrides are shown in [.env.example](./.env.example).
Development credentials are deliberately unsafe; see [environment variables](./docs.md#environment-variables)
before configuring production.

Upgrading an existing installation: the `db` service is now built from
[docker/db](./docker/db/Dockerfile) (same PostgreSQL 16.4/PostGIS 3.4 + pgvector); the existing
`pg_data` volume is reused and migration `0002_cv_pgvector` creates the extension. See
[docs/ml.md §12](./docs/ml.md#12-существующий-том-бд).

## Photo search flow

1. `POST /api/v1/photo-searches/images` (multipart `file`) — private query image.
2. `POST /api/v2/photo-searches` with `query_media_id`, `breed_text`, `coat_color_text`
   (verbatim user text, e.g. "немецкая овчарка + терьер"), optional `bbox`
   (`x_min,y_min,x_max,y_max` in [0,1] on the EXIF-oriented image), `target_kind`, the existing
   filters and `limit` (1–50, default 20).
3. Poll the `Location` URL: `status` keeps the existing job states; `outcome` is `candidates` or
   `rejected` with `reason_code` (`no_dog`, `multiple_dogs`, `invalid_crop`, ...).

Listing photos accept an optional `bbox` form field on upload and
`PUT /api/v1/listings/{id}/photos/{photo_id}/crop`; the listing's `breed`/`color` are the user's
text. The legacy `POST /api/v1/photo-searches` (no breed/colour, limit ≤ 5) still works.

## Verify

Integration tests use a separate Compose project and an isolated PostgreSQL/PostGIS/pgvector
test database:

```sh
docker compose -p pawly-tests --profile test run --build --rm test
docker compose -p pawly-tests --profile test down
```

The test service applies migrations, checks schema drift, runs Ruff, and executes pytest,
including real-weight CV tests (the pinned weights are in the test image).
HTTP smoke uses the running API/worker; in a real CV mode it uploads a public-domain dog photo,
waits for indexing and searches with the same photo (a plumbing check, not a quality check).
It leaves its generated data behind:

```sh
docker compose --profile smoke run --build --rm smoke
```

For persistence, database restart, and migration-failure checks, run from Bash/WSL:

```sh
sh scripts/verify-compose.sh
```

The verification script uses a unique project and retains its named volumes.
Its default port is 18081; choose a free `VERIFY_API_PORT` if needed.
See [development and tests](./docs.md#local-tools-and-tests) for optional host tooling.

## Training experiment

Dataset preparation, head-only training, evaluation and bundle export run in the `ml` profile
(CPU by default), never on `up`:

```sh
docker compose --profile ml run --rm ml python -m app.ml prepare-data oxford-pets
docker compose --profile ml run --rm ml python -m app.ml train --backbone efficientnet_b0 --run-name effb0-s0
docker compose --profile ml run --rm ml python -m app.ml evaluate --run effb0-s0
docker compose --profile ml run --rm ml python -m app.ml export --run effb0-s0 --name effb0-s0-v1
```

Activation, rollback, reindexing and the measured results are in [docs/ml.md](./docs/ml.md).

## Stop and maintain

```sh
docker compose down
```

Named volumes survive this command. **Adding `-v` deletes the database, photos and ML
artifacts.** See [administration, updates, and recovery](./docs.md#administration-updates-and-recovery)
for role management, cleanup, schema updates, and backup/restore.
Local media storage is designed for one Docker host.
