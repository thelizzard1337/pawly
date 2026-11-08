"""Real pretrained weights (no encoder/gate replaced by doubles). Runs in the test image, where
the pinned weights are baked in; skipped elsewhere. Plumbing checks only: two public-domain
photos cannot measure ML quality."""

import io
import os
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from app.core.config import Settings  # noqa: E402
from app.cv import spec  # noqa: E402
from app.modules.media.images import normalize_image  # noqa: E402

pytestmark = pytest.mark.cv_real
FIXTURES = Path(__file__).parent / "fixtures"
WEIGHTS = Path(os.environ.get("CV_WEIGHTS_DIR", "/opt/pawly/weights"))


@pytest.fixture(scope="module")
def extractor():
    from app.cv.extractor import FeatureExtractor
    from app.cv.weights import WeightsError, verify

    try:
        verify("efficientnet_b0", WEIGHTS)
        verify("ssdlite", WEIGHTS)
    except WeightsError:
        pytest.skip("pinned weights are not available here (they are in the Docker test image)")
    return FeatureExtractor.from_settings(
        Settings(_env_file=None, cv_mode="pretrained_features", cv_weights_dir=WEIGHTS)
    )


def canonical(path_or_image) -> bytes:
    if isinstance(path_or_image, Path):
        raw = path_or_image.read_bytes()
    else:
        buffer = io.BytesIO()
        path_or_image.save(buffer, "JPEG", quality=95)
        raw = buffer.getvalue()
    return normalize_image(raw, 20_000_000).content


def test_real_encoder_and_gate_accept_single_dogs(extractor):
    beagle = extractor.extract(canonical(FIXTURES / "dog_beagle_pd.jpg"), None)
    labrador = extractor.extract(canonical(FIXTURES / "dog_labrador_pd.jpg"), None)
    for result in (beagle, labrador):
        assert result.rejected is None
        assert len(result.embedding) == spec.BACKBONES["efficientnet_b0"].dimension == 1280
        assert sum(v * v for v in result.embedding) == pytest.approx(1.0, abs=1e-4)
        assert result.breed_prediction is None  # pretrained_features: no heads
    again = extractor.extract(canonical(FIXTURES / "dog_beagle_pd.jpg"), None)
    assert again.embedding == pytest.approx(beagle.embedding, abs=1e-6)  # deterministic
    cosine = sum(a * b for a, b in zip(beagle.embedding, labrador.embedding, strict=True))
    assert cosine < 0.999  # different photos are not collapsed to one vector


def test_real_gate_rejects_no_dog_and_multiple_dogs(extractor):
    solid = extractor.extract(canonical(Image.new("RGB", (320, 320), (120, 80, 40))), None)
    assert solid.rejected == spec.NO_DOG and solid.embedding is None
    beagle = Image.open(FIXTURES / "dog_beagle_pd.jpg").convert("RGB").resize((300, 450))
    labrador = Image.open(FIXTURES / "dog_labrador_pd.jpg").convert("RGB").resize((450, 300))
    collage = Image.new("RGB", (760, 460), "white")
    collage.paste(beagle, (0, 0))
    collage.paste(labrador, (305, 80))
    raw = canonical(collage)
    assert extractor.extract(raw, None).rejected == spec.MULTIPLE_DOGS
    # The user crop decides what is checked: cropping to one dog passes the gate.
    left = extractor.extract(raw, spec.BBox(0, 0, 0.4, 1))
    assert left.rejected is None and left.crop.box == (0, 0, 304, 460)
