#!/bin/sh
set -e
# Runs once, after 10_postgis.sh, only when the data directory is initialized for the first time.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;"
