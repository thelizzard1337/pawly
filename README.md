# PawLy — backend MVP

Backend API for lost/found animal listings and shelter adoption.
The API includes accounts, private contacts, photos, geographic filters, map points,
shelter verification, adoption applications, contact requests, and asynchronous mock photo search.
There is no frontend.

**The photo-search mock does not measure visual similarity.** It returns real eligible listings
in deterministic SHA-256 order with `similarity_score=null`, `provider="mock-v1"`,
`is_mock=true`, and `model_version=null`.

The canonical technical reference and architectural decision log are in [docs.md](./docs.md).
Configuration, authentication, data flows, maintenance, backup/restore, conventions, and known
limitations are maintained there.

## Start

Docker Engine and Docker Compose v2 are sufficient. Run from the repository root:

```sh
docker compose up
```

Compose builds the runtime image if it is absent, waits for PostgreSQL/PostGIS, applies
Alembic migrations, then starts API and worker. No host Python, uv, mandatory `.env`,
manual migration, or preparation script is required. The first build requires network access.

- API: http://localhost:8000/api/v1
- Swagger: http://localhost:8000/docs
- OpenAPI: http://localhost:8000/openapi.json
- Liveness: http://localhost:8000/health/live
- Readiness: http://localhost:8000/health/ready

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

## Verify

Integration tests use a separate Compose project and an isolated PostgreSQL/PostGIS test database:

```sh
docker compose -p pawly-tests --profile test run --build --rm test
docker compose -p pawly-tests --profile test down
```

The test service applies migrations, checks schema drift, runs Ruff, and executes pytest.
HTTP smoke uses running API/worker services and leaves its generated data behind:

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

## Stop and maintain

```sh
docker compose down
```

Named volumes survive this command. **Adding `-v` deletes the database and photos.**
See [administration, updates, and recovery](./docs.md#administration-updates-and-recovery)
for role management, cleanup, schema updates, and backup/restore.
Local media storage is designed for one Docker host; real CV remains unimplemented research.
