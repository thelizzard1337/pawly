import io
from datetime import timedelta
from uuid import uuid4

import jwt
import pytest
from PIL import Image
from pydantic import ValidationError

from app.api.pagination import decode_cursor, encode_cursor
from app.core.config import DEV_SECRET, Settings
from app.core.security import access_token, decode_access, utcnow
from app.integrations.cv.matcher import Candidate, MockPhotoMatcher
from app.integrations.storage.local import LocalFileStorage
from app.modules.listings.schemas import Filters, ListingCreate
from app.modules.media.images import InvalidImage, normalize_image


def test_production_rejects_unsafe_configuration():
    for settings in (
        {"app_env": "production"},
        {"cv_mode": "real"},
        {"cors_origins": ["*"]},
        {"worker_lease_seconds": 10},
    ):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **settings)
    value = Settings(
        _env_file=None,
        app_env="production",
        jwt_secret="s" * 48,
        db_password="strong-db-password",
        cookie_secure=True,
        cors_origins=["https://pawly.example.com"],
    )
    assert value.cookie_secure
    assert DEV_SECRET != value.jwt_secret


def test_database_url_escapes_password():
    settings = Settings(_env_file=None, db_password="p@ss:/?#%word")
    assert settings.database_url.password == "p@ss:/?#%word"
    assert "p@ss:" not in settings.database_url.render_as_string(hide_password=False)


def test_jwt_claim_validation():
    settings, uid = Settings(_env_file=None), uuid4()
    assert decode_access(access_token(uid, settings), settings) == uid
    valid = {
        "sub": str(uid),
        "exp": utcnow() + timedelta(minutes=1),
        "iat": utcnow(),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
    }
    for patch in ({"exp": utcnow() - timedelta(seconds=1)}, {"aud": "other"}, {"iss": "other"}):
        token = jwt.encode({**valid, **patch}, settings.jwt_secret, algorithm="HS256")
        with pytest.raises(jwt.PyJWTError):
            decode_access(token, settings)
    for missing in ("exp", "sub", "iss", "aud"):
        payload = {key: value for key, value in valid.items() if key != missing}
        with pytest.raises(jwt.PyJWTError):
            decode_access(jwt.encode(payload, settings.jwt_secret, algorithm="HS256"), settings)
    with pytest.raises(jwt.PyJWTError):
        decode_access(jwt.encode(valid, settings.jwt_secret, algorithm="HS384"), settings)


def test_image_normalization_removes_exif_and_orientation():
    image = Image.new("RGB", (30, 20), "green")
    exif = Image.Exif()
    exif[274] = 6
    exif[34853] = {1: "N", 2: (43, 0, 0), 3: "E", 4: (76, 0, 0)}
    raw = io.BytesIO()
    image.save(raw, "JPEG", exif=exif)
    result = normalize_image(raw.getvalue(), 20_000_000)
    assert (result.width, result.height) == (20, 30)
    with Image.open(io.BytesIO(result.content)) as final:
        assert not final.getexif()
        assert "exif" not in final.info


def test_invalid_animated_and_large_images(image):
    with pytest.raises(InvalidImage):
        normalize_image(b"not an image", 20_000_000)
    with pytest.raises(InvalidImage):
        normalize_image(image, 10)
    raw = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(
        raw, "PNG", save_all=True, append_images=[Image.new("RGB", (8, 8), "blue")]
    )
    with pytest.raises(InvalidImage):
        normalize_image(raw.getvalue(), 20_000_000)


def test_storage_atomic_and_path_safety(tmp_path):
    storage = LocalFileStorage(tmp_path)
    storage.check()
    key = f"{uuid4()}.jpg"
    storage.write(key, b"content")
    assert storage.path(key).read_bytes() == b"content"
    assert not storage.path(key).with_suffix(".tmp").exists()
    storage.delete(key)
    storage.delete(key)
    for invalid in ("../secret.jpg", "C:/secret.jpg", "bad.jpg", f"{uuid4()}.jpg/../x"):
        with pytest.raises(ValueError):
            storage.path(invalid)


@pytest.mark.parametrize(
    "params",
    [
        {"latitude": 30},
        {"latitude": 91, "longitude": 30, "radius_m": 1},
        {"latitude": 30, "longitude": 181, "radius_m": 1},
        {"latitude": 30, "longitude": 30, "radius_m": -1},
        {"latitude": 30, "longitude": 30, "radius_m": 100001},
        {"event_from": "2026-11-06T10:00:00"},
        {"event_from": "2026-11-06T10:00:00Z", "event_to": "2026-11-05T10:00:00Z"},
    ],
)
def test_invalid_filters(params):
    with pytest.raises(ValidationError):
        Filters(**params)


def test_kind_shelter_invariant_and_cursor():
    with pytest.raises(ValidationError):
        ListingCreate(kind="adoption")
    with pytest.raises(ValidationError):
        ListingCreate(kind="lost", shelter_id=uuid4())
    from types import SimpleNamespace

    row = SimpleNamespace(id=uuid4(), created_at=utcnow())
    assert decode_cursor(encode_cursor(row)) == (row.created_at, row.id)


async def test_mock_is_deterministic_unique_and_bounded():
    matcher = MockPhotoMatcher()
    candidates = [Candidate(uuid4(), uuid4()) for _ in range(10)]
    candidates += [Candidate(candidates[0].listing_id, uuid4())]
    first = await matcher.match("query-hash", candidates, 5)
    second = await matcher.match("query-hash", list(reversed(candidates)), 5)
    assert first == second
    assert len({item.listing_id for item in first}) == 5
    assert await matcher.match("query-hash", [], 5) == []


def test_cursor_rejects_malformed_structure():
    import base64
    import json

    from fastapi import HTTPException

    for payload in ([utcnow().isoformat(), {}], [1, 2], {}, None, ["bad", "bad"]):
        encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        with pytest.raises(HTTPException) as error:
            decode_cursor(encoded)
        assert error.value.status_code == 422
