#!/bin/sh
# Optional full infrastructure verification. Requires Docker Compose v2 and a POSIX shell.
set -eu
cd "$(dirname "$0")/.."
project="pawly-check-$(date +%s)-$$"
export API_PORT="${VERIFY_API_PORT:-18081}"
override="$(mktemp)"
dc() { docker compose --env-file /dev/null -p "$project" "$@"; }
finish() {
    dc down --remove-orphans
    rm -f "$override"
    echo "Verification project: $project (its named volumes are retained)"
}
trap finish EXIT
dc config --quiet
# No --build is necessary for initial startup; Compose builds if the image is absent.
dc up -d --wait api worker
dc --profile test run --build --rm test
dc --profile smoke build smoke
snapshot="$(dc --profile smoke run --rm --no-deps -T smoke \
    python -m scripts.persistence capture http://api:8000)"
dc up -d --force-recreate --wait db api worker
printf '%s' "$snapshot" | dc --profile smoke run --rm --no-deps -T smoke \
    python -m scripts.persistence check http://api:8000
dc restart db
dc up -d --wait db
dc --profile smoke run --rm --no-deps -T smoke
echo "API and worker recovered after database restart"
dc down --remove-orphans
cat > "$override" <<'YAML'
services:
  migrate:
    command: ["alembic", "upgrade", "missing_revision_for_verification"]
YAML
if docker compose --env-file /dev/null -p "$project" -f compose.yaml -f "$override" \
    up -d --wait api worker; then
    echo "ERROR: a failed migration did not block startup" >&2
    exit 1
fi
test -z "$(dc ps --status running -q api worker)"
echo "Migration failure correctly blocked API and worker startup"
dc up -d --wait api worker
echo "Full infrastructure verification passed"
