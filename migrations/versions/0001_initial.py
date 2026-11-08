from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

STATEMENTS = [
    """
CREATE EXTENSION IF NOT EXISTS postgis
""",
    """
CREATE TABLE shelters (
        name VARCHAR(200) NOT NULL,
        description TEXT NOT NULL,
        city VARCHAR(200) NOT NULL,
        address_public VARCHAR(300) NOT NULL,
        verification_status VARCHAR(20) NOT NULL,
        id UUID NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_shelters PRIMARY KEY (id),
        CONSTRAINT ck_shelters_verification CHECK (verification_status IN
('pending','verified','rejected'))
)
""",
    """
CREATE TABLE users (
        email VARCHAR(320) NOT NULL,
        password_hash TEXT NOT NULL,
        display_name VARCHAR(100) NOT NULL,
        is_admin BOOLEAN DEFAULT false NOT NULL,
        is_active BOOLEAN DEFAULT true NOT NULL,
        id UUID NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_users PRIMARY KEY (id),
        CONSTRAINT uq_users_email UNIQUE (email)
)
""",
    """
CREATE TABLE listings (
        author_id UUID NOT NULL,
        shelter_id UUID,
        kind VARCHAR(20) NOT NULL,
        status VARCHAR(20) NOT NULL,
        is_hidden BOOLEAN DEFAULT false NOT NULL,
        species VARCHAR(60),
        sex VARCHAR(30),
        color VARCHAR(100),
        breed VARCHAR(100),
        title VARCHAR(200),
        description TEXT,
        location geography(Point,4326),
        location_label VARCHAR(300),
        event_at TIMESTAMP WITH TIME ZONE,
        id UUID NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_listings PRIMARY KEY (id),
        CONSTRAINT ck_listings_kind CHECK (kind IN ('lost','found','adoption')),
        CONSTRAINT ck_listings_status CHECK (status IN
('draft','active','resolved','archived')),
        CONSTRAINT ck_listings_kind_shelter CHECK ((kind = 'adoption') = (shelter_id IS NOT
NULL)),
        CONSTRAINT ck_listings_active_complete CHECK (status <> 'active' OR (species IS NOT
NULL AND length(trim(species)) > 0 AND location IS NOT NULL AND description IS NOT NULL
AND length(trim(description)) > 0 AND (kind='adoption' OR event_at IS NOT NULL))),
        CONSTRAINT fk_listings_author_id FOREIGN KEY(author_id) REFERENCES users (id),
        CONSTRAINT fk_listings_shelter_id FOREIGN KEY(shelter_id) REFERENCES shelters (id)
)
""",
    """
CREATE TABLE refresh_sessions (
        user_id UUID NOT NULL,
        family_id UUID NOT NULL,
        token_hash VARCHAR(64) NOT NULL,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
        revoked_at TIMESTAMP WITH TIME ZONE,
        replaced_by_id UUID,
        id UUID NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_refresh_sessions PRIMARY KEY (id),
        CONSTRAINT fk_refresh_sessions_user_id FOREIGN KEY(user_id) REFERENCES users (id),
        CONSTRAINT uq_refresh_sessions_token_hash UNIQUE (token_hash),
        CONSTRAINT fk_refresh_sessions_replaced_by_id FOREIGN KEY(replaced_by_id) REFERENCES
refresh_sessions (id)
)
""",
    """
CREATE TABLE shelter_members (
        shelter_id UUID NOT NULL,
        user_id UUID NOT NULL,
        role VARCHAR(20) NOT NULL,
        CONSTRAINT pk_shelter_members PRIMARY KEY (shelter_id, user_id),
        CONSTRAINT ck_shelter_members_role CHECK (role = 'manager'),
        CONSTRAINT fk_shelter_members_shelter_id FOREIGN KEY(shelter_id) REFERENCES shelters
(id),
        CONSTRAINT fk_shelter_members_user_id FOREIGN KEY(user_id) REFERENCES users (id)
)
""",
    """
CREATE TABLE user_contacts (
        user_id UUID NOT NULL,
        phone VARCHAR(40),
        preferred_channel VARCHAR(100),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_user_contacts PRIMARY KEY (user_id),
        CONSTRAINT fk_user_contacts_user_id FOREIGN KEY(user_id) REFERENCES users (id)
)
""",
    """
CREATE TABLE adoption_requests (
        listing_id UUID NOT NULL,
        applicant_id UUID NOT NULL,
        message TEXT NOT NULL,
        status VARCHAR(20) NOT NULL,
        reviewed_by UUID,
        reviewed_at TIMESTAMP WITH TIME ZONE,
        id UUID NOT NULL,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_adoption_requests PRIMARY KEY (id),
        CONSTRAINT ck_adoption_requests_status CHECK (status IN
('pending','approved','rejected','withdrawn')),
        CONSTRAINT fk_adoption_requests_listing_id FOREIGN KEY(listing_id) REFERENCES
listings (id),
        CONSTRAINT fk_adoption_requests_applicant_id FOREIGN KEY(applicant_id) REFERENCES
users (id),
        CONSTRAINT fk_adoption_requests_reviewed_by FOREIGN KEY(reviewed_by) REFERENCES
users (id)
)
""",
    """
CREATE TABLE contact_requests (
        listing_id UUID NOT NULL,
        sender_id UUID NOT NULL,
        message TEXT NOT NULL,
        status VARCHAR(20) NOT NULL,
        id UUID NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_contact_requests PRIMARY KEY (id),
        CONSTRAINT ck_contact_requests_status CHECK (status='pending'),
        CONSTRAINT fk_contact_requests_listing_id FOREIGN KEY(listing_id) REFERENCES
listings (id),
        CONSTRAINT fk_contact_requests_sender_id FOREIGN KEY(sender_id) REFERENCES users
(id)
)
""",
    """
CREATE TABLE media_assets (
        owner_id UUID NOT NULL,
        listing_id UUID,
        purpose VARCHAR(20) NOT NULL,
        status VARCHAR(20) NOT NULL,
        storage_key VARCHAR(100) NOT NULL,
        sha256 VARCHAR(64),
        content_type VARCHAR(50),
        size_bytes INTEGER,
        width INTEGER,
        height INTEGER,
        position INTEGER,
        expires_at TIMESTAMP WITH TIME ZONE,
        id UUID NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_media_assets PRIMARY KEY (id),
        CONSTRAINT ck_media_assets_purpose CHECK (purpose IN ('listing','search')),
        CONSTRAINT ck_media_assets_status CHECK (status IN
('pending','ready','failed','deleted')),
        CONSTRAINT ck_media_assets_attachment CHECK ((purpose='listing' AND listing_id IS
NOT NULL AND position IS NOT NULL AND position >= 0) OR (purpose='search'
AND listing_id IS NULL AND position IS NULL)),
        CONSTRAINT ck_media_assets_ready_metadata CHECK (status <> 'ready' OR (sha256 IS NOT
NULL AND content_type IS NOT NULL AND size_bytes IS NOT NULL AND width IS NOT NULL
AND height IS NOT NULL AND size_bytes > 0 AND width > 0 AND height > 0)),
        CONSTRAINT fk_media_assets_owner_id FOREIGN KEY(owner_id) REFERENCES users (id),
        CONSTRAINT fk_media_assets_listing_id FOREIGN KEY(listing_id) REFERENCES listings
(id),
        CONSTRAINT uq_media_assets_storage_key UNIQUE (storage_key)
)
""",
    """
CREATE TABLE photo_searches (
        owner_id UUID NOT NULL,
        query_media_id UUID NOT NULL,
        scope JSONB NOT NULL,
        "limit" INTEGER NOT NULL,
        status VARCHAR(20) NOT NULL,
        provider VARCHAR(50) NOT NULL,
        model_version VARCHAR(100),
        attempts INTEGER NOT NULL,
        available_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        lease_token UUID,
        lease_until TIMESTAMP WITH TIME ZONE,
        error_code VARCHAR(100),
        idempotency_key VARCHAR(200),
        request_fingerprint VARCHAR(64),
        started_at TIMESTAMP WITH TIME ZONE,
        finished_at TIMESTAMP WITH TIME ZONE,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
        id UUID NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
        CONSTRAINT pk_photo_searches PRIMARY KEY (id),
        CONSTRAINT ck_photo_searches_status CHECK (status IN
('queued','running','completed','failed')),
        CONSTRAINT ck_photo_searches_limit CHECK ("limit" BETWEEN 1 AND 5),
        CONSTRAINT ck_photo_searches_attempts CHECK (attempts >= 0),
        CONSTRAINT ck_photo_searches_running_lease CHECK (status <> 'running' OR
(lease_token IS NOT NULL AND lease_until IS NOT NULL)),
        CONSTRAINT fk_photo_searches_owner_id FOREIGN KEY(owner_id) REFERENCES users (id),
        CONSTRAINT fk_photo_searches_query_media_id FOREIGN KEY(query_media_id) REFERENCES
media_assets (id)
)
""",
    """
CREATE TABLE photo_search_results (
        search_id UUID NOT NULL,
        listing_id UUID NOT NULL,
        matched_media_id UUID NOT NULL,
        rank INTEGER NOT NULL,
        similarity_score DOUBLE PRECISION,
        CONSTRAINT pk_photo_search_results PRIMARY KEY (search_id, listing_id),
        CONSTRAINT uq_photo_search_results_search_id UNIQUE (search_id, rank),
        CONSTRAINT ck_photo_search_results_rank CHECK (rank BETWEEN 1 AND 5),
        CONSTRAINT ck_photo_search_results_mock_score CHECK (similarity_score IS NULL),
        CONSTRAINT fk_photo_search_results_search_id FOREIGN KEY(search_id) REFERENCES
photo_searches (id) ON DELETE CASCADE,
        CONSTRAINT fk_photo_search_results_listing_id FOREIGN KEY(listing_id) REFERENCES
listings (id),
        CONSTRAINT fk_photo_search_results_matched_media_id FOREIGN KEY(matched_media_id)
REFERENCES media_assets (id)
)
""",
    """
CREATE INDEX ix_listings_author_id ON listings (author_id)
""",
    """
CREATE INDEX ix_listings_cursor ON listings (created_at, id)
""",
    """
CREATE INDEX ix_listings_filters ON listings (kind, status, species, created_at)
""",
    """
CREATE INDEX ix_listings_location ON listings USING gist (location)
""",
    """
CREATE INDEX ix_listings_shelter_id ON listings (shelter_id)
""",
    """
CREATE INDEX ix_refresh_sessions_expires_at ON refresh_sessions (expires_at)
""",
    """
CREATE INDEX ix_refresh_sessions_family_id ON refresh_sessions (family_id)
""",
    """
CREATE INDEX ix_refresh_sessions_user_id ON refresh_sessions (user_id)
""",
    """
CREATE INDEX ix_shelter_members_user_id ON shelter_members (user_id)
""",
    """
CREATE INDEX ix_adoption_applicant_created ON adoption_requests (applicant_id, created_at)
""",
    """
CREATE INDEX ix_adoption_listing_status ON adoption_requests (listing_id, status)
""",
    """
CREATE UNIQUE INDEX uq_adoption_approved ON adoption_requests (listing_id) WHERE
status='approved'
""",
    """
CREATE UNIQUE INDEX uq_adoption_pending ON adoption_requests (listing_id, applicant_id)
WHERE status='pending'
""",
    """
CREATE INDEX ix_contact_requests_listing_id ON contact_requests (listing_id)
""",
    """
CREATE INDEX ix_contact_requests_sender_id ON contact_requests (sender_id)
""",
    """
CREATE INDEX ix_media_assets_expires_at ON media_assets (expires_at)
""",
    """
CREATE INDEX ix_media_assets_listing_id ON media_assets (listing_id)
""",
    """
CREATE INDEX ix_media_owner_created ON media_assets (owner_id, created_at)
""",
    """
CREATE INDEX ix_media_status_created ON media_assets (status, created_at)
""",
    """
CREATE UNIQUE INDEX uq_media_position ON media_assets (listing_id, position) WHERE status IN
('pending','ready')
""",
    """
CREATE INDEX ix_photo_searches_expires_at ON photo_searches (expires_at)
""",
    """
CREATE INDEX ix_photo_searches_query_media_id ON photo_searches (query_media_id)
""",
    """
CREATE INDEX ix_search_claim ON photo_searches (status, available_at)
""",
    """
CREATE INDEX ix_search_lease ON photo_searches (status, lease_until)
""",
    """
CREATE INDEX ix_search_owner_created ON photo_searches (owner_id, created_at)
""",
    """
CREATE UNIQUE INDEX uq_search_idempotency ON photo_searches (owner_id, idempotency_key)
WHERE idempotency_key IS NOT NULL AND idempotency_key <> ''
""",
]


def upgrade():
    for statement in STATEMENTS:
        op.execute(statement)


def downgrade():
    for table in [
        "photo_search_results",
        "photo_searches",
        "media_assets",
        "contact_requests",
        "adoption_requests",
        "user_contacts",
        "shelter_members",
        "refresh_sessions",
        "listings",
        "users",
        "shelters",
    ]:
        op.drop_table(table)
