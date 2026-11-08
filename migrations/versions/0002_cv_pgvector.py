"""CV features (pgvector), indexing queue, v2 photo-search fields, CV worker heartbeat.

Existing rows are preserved: legacy searches get api_version='v1' and NULL user text.
"""

from alembic import op
from sqlalchemy import text

revision = "0002_cv_pgvector"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def ensure_pgvector():
    """Create pgvector only when needed and allowed; never hide a missing extension."""
    bind = op.get_bind()
    if bind.scalar(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")):
        return
    if not bind.scalar(text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")):
        raise RuntimeError(
            "pgvector is not installed in the PostgreSQL server image. Rebuild/pull the db "
            "service image (docker compose build db) and restart db; see docs/ml.md."
        )
    # pgvector 0.8.0 is not a "trusted" extension: creating it needs a superuser. Fresh Compose
    # volumes get it from the db image init script; this branch covers older volumes.
    superuser = bind.scalar(text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user"))
    if not superuser:
        raise RuntimeError(
            "pgvector is available but not created, and the migration role is not a superuser. "
            "Ask a DBA to run once: CREATE EXTENSION IF NOT EXISTS vector; then rerun the "
            "migration."
        )
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


STATEMENTS = [
    """
ALTER TABLE media_assets ADD COLUMN crop_bbox DOUBLE PRECISION[]
""",
    """
CREATE TABLE media_features (
        media_id UUID NOT NULL,
        purpose VARCHAR(20) NOT NULL,
        status VARCHAR(20) NOT NULL,
        reason_code VARCHAR(40),
        error_code VARCHAR(100),
        pipeline_id VARCHAR(200) NOT NULL,
        embedding_space_id VARCHAR(120) NOT NULL,
        cv_mode VARCHAR(30) NOT NULL,
        backbone VARCHAR(50) NOT NULL,
        weights_id VARCHAR(50) NOT NULL,
        preprocessing_id VARCHAR(50) NOT NULL,
        descriptor_version VARCHAR(50) NOT NULL,
        gate_id VARCHAR(80) NOT NULL,
        bundle_id VARCHAR(120),
        dimension INTEGER NOT NULL,
        crop_rule VARCHAR(20) NOT NULL,
        crop_bbox DOUBLE PRECISION[],
        crop_box INTEGER[],
        crop_hash VARCHAR(64) NOT NULL,
        input_sha256 VARCHAR(64) NOT NULL,
        embedding vector,
        rgb_histogram DOUBLE PRECISION[],
        dominant_colors JSONB,
        breed_prediction JSONB,
        color_prediction DOUBLE PRECISION[],
        extraction_ms DOUBLE PRECISION,
        user_breed_text TEXT,
        user_color_text TEXT,
        attempts INTEGER NOT NULL,
        available_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        lease_token UUID,
        lease_until TIMESTAMP WITH TIME ZONE,
        indexed_at TIMESTAMP WITH TIME ZONE,
        superseded_at TIMESTAMP WITH TIME ZONE,
        id UUID NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_media_features PRIMARY KEY (id),
        CONSTRAINT ck_media_features_purpose CHECK (purpose IN ('listing','query')),
        CONSTRAINT ck_media_features_status CHECK (status IN
('pending','running','indexed','rejected','failed')),
        CONSTRAINT ck_media_features_attempts CHECK (attempts >= 0),
        CONSTRAINT ck_media_features_running_lease CHECK (status <> 'running' OR
(lease_token IS NOT NULL AND lease_until IS NOT NULL)),
        CONSTRAINT ck_media_features_indexed_complete CHECK (status <> 'indexed' OR
(embedding IS NOT NULL AND vector_dims(embedding) = dimension AND rgb_histogram IS NOT NULL
AND cardinality(rgb_histogram) = 64)),
        CONSTRAINT ck_media_features_rejected_reason CHECK ((status = 'rejected') =
(reason_code IS NOT NULL) AND (reason_code IS NULL OR reason_code IN
('no_dog','multiple_dogs','invalid_crop','invalid_image','unsupported_image'))),
        CONSTRAINT ck_media_features_dimension CHECK (dimension > 0),
        CONSTRAINT uq_media_features_media_id UNIQUE (media_id, crop_hash, pipeline_id),
        CONSTRAINT fk_media_features_media_id FOREIGN KEY(media_id) REFERENCES media_assets (id)
)
""",
    """
CREATE INDEX ix_media_features_media_id ON media_features (media_id)
""",
    """
CREATE INDEX ix_media_features_claim ON media_features (pipeline_id, status, available_at)
""",
    """
CREATE UNIQUE INDEX uq_media_features_current ON media_features (media_id, pipeline_id)
WHERE purpose = 'listing' AND superseded_at IS NULL
""",
    """
CREATE TABLE cv_worker_status (
        worker_id VARCHAR(100) NOT NULL,
        cv_mode VARCHAR(30) NOT NULL,
        provider VARCHAR(50) NOT NULL,
        state VARCHAR(20) NOT NULL,
        pipeline_id VARCHAR(200),
        embedding_space_id VARCHAR(120),
        bundle_id VARCHAR(120),
        error_code VARCHAR(200),
        details JSONB,
        started_at TIMESTAMP WITH TIME ZONE NOT NULL,
        heartbeat_at TIMESTAMP WITH TIME ZONE NOT NULL,
        CONSTRAINT pk_cv_worker_status PRIMARY KEY (worker_id),
        CONSTRAINT ck_cv_worker_status_state CHECK (state IN ('loading','ready','error'))
)
""",
    """
ALTER TABLE photo_searches
    ADD COLUMN api_version VARCHAR(10) DEFAULT 'v1' NOT NULL,
    ADD COLUMN breed_text TEXT,
    ADD COLUMN coat_color_text TEXT,
    ADD COLUMN breed_text_norm TEXT,
    ADD COLUMN coat_color_text_norm TEXT,
    ADD COLUMN crop_bbox DOUBLE PRECISION[],
    ADD COLUMN outcome VARCHAR(20),
    ADD COLUMN reason_code VARCHAR(40),
    ADD COLUMN cv_mode VARCHAR(30),
    ADD COLUMN pipeline_id VARCHAR(200),
    ADD COLUMN embedding_space_id VARCHAR(120),
    ADD COLUMN fusion_version VARCHAR(50),
    ADD COLUMN query_feature_id UUID,
    ADD COLUMN search_meta JSONB,
    ADD CONSTRAINT fk_photo_searches_query_feature_id FOREIGN KEY(query_feature_id)
        REFERENCES media_features (id) ON DELETE SET NULL,
    DROP CONSTRAINT ck_photo_searches_limit,
    ADD CONSTRAINT ck_photo_searches_limit CHECK ("limit" BETWEEN 1 AND 50),
    ADD CONSTRAINT ck_photo_searches_api_version CHECK (api_version IN ('v1','v2')),
    ADD CONSTRAINT ck_photo_searches_legacy_limit CHECK (api_version <> 'v1' OR "limit" <= 5),
    ADD CONSTRAINT ck_photo_searches_v2_user_text CHECK (api_version <> 'v2' OR
(breed_text IS NOT NULL AND length(trim(breed_text)) > 0 AND coat_color_text IS NOT NULL
AND length(trim(coat_color_text)) > 0)),
    ADD CONSTRAINT ck_photo_searches_outcome CHECK (outcome IS NULL OR outcome IN
('candidates','rejected')),
    ADD CONSTRAINT ck_photo_searches_outcome_reason CHECK ((outcome = 'rejected') =
(reason_code IS NOT NULL))
""",
    """
ALTER TABLE photo_search_results
    ADD COLUMN ranking_score DOUBLE PRECISION,
    DROP CONSTRAINT ck_photo_search_results_rank,
    ADD CONSTRAINT ck_photo_search_results_rank CHECK (rank BETWEEN 1 AND 50)
""",
]


def upgrade():
    ensure_pgvector()
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade():
    # The vector extension is left installed: other objects may depend on it.
    op.execute(
        """
ALTER TABLE photo_search_results
    DROP CONSTRAINT ck_photo_search_results_rank,
    DROP COLUMN ranking_score
"""
    )
    op.execute("DELETE FROM photo_search_results WHERE rank > 5")
    op.execute(
        "ALTER TABLE photo_search_results ADD CONSTRAINT ck_photo_search_results_rank "
        "CHECK (rank BETWEEN 1 AND 5)"
    )
    op.execute(
        """
ALTER TABLE photo_searches
    DROP CONSTRAINT ck_photo_searches_limit,
    DROP CONSTRAINT fk_photo_searches_query_feature_id,
    DROP COLUMN api_version, DROP COLUMN breed_text, DROP COLUMN coat_color_text,
    DROP COLUMN breed_text_norm, DROP COLUMN coat_color_text_norm, DROP COLUMN crop_bbox,
    DROP COLUMN outcome, DROP COLUMN reason_code, DROP COLUMN cv_mode, DROP COLUMN pipeline_id,
    DROP COLUMN embedding_space_id, DROP COLUMN fusion_version, DROP COLUMN query_feature_id,
    DROP COLUMN search_meta
"""
    )
    op.execute('UPDATE photo_searches SET "limit" = 5 WHERE "limit" > 5')
    op.execute(
        'ALTER TABLE photo_searches ADD CONSTRAINT ck_photo_searches_limit CHECK ("limit" '
        "BETWEEN 1 AND 5)"
    )
    op.drop_table("cv_worker_status")
    op.drop_table("media_features")
    op.execute("ALTER TABLE media_assets DROP COLUMN crop_bbox")
