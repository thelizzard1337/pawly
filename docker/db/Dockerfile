# PostgreSQL 16 + PostGIS 3.4 (unchanged upstream base) + pgvector for CV embeddings.
# Same PostgreSQL server packages as before: an existing pg_data volume stays compatible.
FROM postgis/postgis:16-3.4
# The bullseye PGDG suite moved to the official PGDG archive; the upstream signing key is reused.
# dpkg installs only the extension, so apt cannot upgrade the PostgreSQL/PostGIS server packages.
ARG PGVECTOR_DEB_VERSION=0.8.0-1.pgdg110+1
RUN echo "deb [ signed-by=/usr/local/share/keyrings/postgres.gpg.asc ] http://apt-archive.postgresql.org/pub/repos/apt bullseye-pgdg-archive main" \
        > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && cd /tmp && apt-get download "postgresql-16-pgvector=${PGVECTOR_DEB_VERSION}" \
    && dpkg -i /tmp/postgresql-16-pgvector_*.deb \
    && rm -f /tmp/postgresql-16-pgvector_*.deb \
    && rm -rf /var/lib/apt/lists/* \
    && test -f /usr/share/postgresql/16/extension/vector.control
# Fresh volumes only: create the extension as the bootstrap superuser, so the migration does not
# need superuser rights later. Existing volumes: see docs/ml.md ("Existing database volume").
COPY 20-pgvector.sh /docker-entrypoint-initdb.d/20-pgvector.sh
