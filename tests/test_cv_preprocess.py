"""Crop rules, RGB descriptor, text normalisation, metrics. No torch required."""

import io
import math

import pytest
from PIL import Image

from app.core.config import Settings
from app.cv import spec
from app.cv.text import normalize_text
from app.modules.media.images import normalize_image

np = pytest.importorskip("numpy")
image_ops = pytest.importorskip("app.cv.image_ops")


def canonical(image: Image.Image, fmt="PNG", **save) -> bytes:
    raw = io.BytesIO()
    image.save(raw, fmt, **save)
    return normalize_image(raw.getvalue(), 20_000_000).content


@pytest.mark.parametrize(
    "values",
    [
        (0, 0, 0, 1),  # zero width
        (0.5, 0, 0.4, 1),  # inverted
        (-0.1, 0, 1, 1),
        (0, 0, 1, 1.01),
        (math.nan, 0, 1, 1),
        (math.inf, 0, 1, 1),
    ],
)
def test_bbox_rejects_invalid_values(values):
    with pytest.raises(spec.InvalidCrop):
        spec.BBox(*values)


def test_bbox_parse_and_rounding_rule_v1():
    bbox = spec.BBox.parse(" 0.1, 0.2 ,0.3,0.4")
    crop = spec.crop_spec(bbox, 1000, 500)
    # floor for left/top, ceil for right/bottom after rounding value*size to 6 decimals:
    # 0.3 * 1000 = 300.00000000000006 must not become 301.
    assert crop.box == (100, 100, 300, 200)
    assert crop.rule == "crop-v1" and crop.bbox == [0.1, 0.2, 0.3, 0.4]
    assert spec.crop_spec(spec.BBox(0.1001, 0, 0.5, 1), 1000, 500).box == (100, 0, 500, 500)
    assert spec.crop_spec(spec.BBox(0, 0, 0.5004, 1), 1000, 500).box == (0, 0, 501, 500)
    full = spec.crop_spec(None, 640, 480)
    assert full.box == (0, 0, 640, 480) and full.bbox is None
    assert full.hash != crop.hash
    assert spec.crop_spec(None, 640, 480).hash == full.hash  # deterministic
    for raw in ("1,2,3", "a,b,c,d", "0,0,1,1,1", "x" * 300):
        with pytest.raises(spec.InvalidCrop):
            spec.BBox.parse(raw)


def test_crop_minimum_side_and_clamping():
    with pytest.raises(spec.InvalidCrop):
        spec.crop_spec(spec.BBox(0, 0, 0.01, 1), 1000, 1000)  # 10 px wide
    assert spec.crop_spec(spec.BBox(0.4, 0, 0.9999999999, 1), 64, 64).box == (25, 0, 64, 64)
    with pytest.raises(spec.InvalidCrop):
        spec.crop_spec(None, 20, 20)


def test_crop_is_applied_after_exif_orientation():
    # Left half red, right half blue; EXIF orientation 6 rotates 90 degrees clockwise, so on
    # the canonical image the red half is on TOP.
    image = Image.new("RGB", (200, 100), "blue")
    image.paste(Image.new("RGB", (100, 100), "red"), (0, 0))
    exif = Image.Exif()
    exif[274] = 6
    raw = io.BytesIO()
    image.save(raw, "JPEG", exif=exif, quality=95)
    canonical_bytes = normalize_image(raw.getvalue(), 20_000_000).content
    loaded = image_ops.load_canonical(canonical_bytes)
    assert loaded.size == (100, 200)
    crop, crop_spec = image_ops.apply_crop(loaded, spec.BBox(0, 0, 1, 0.45))
    assert crop_spec.box == (0, 0, 100, 90)
    mean = np.asarray(crop, dtype=np.float64).reshape(-1, 3).mean(0)
    assert mean[0] > 200 and mean[2] < 60  # red, not blue


def test_rgb_histogram_is_computed_on_raw_crop_pixels():
    image = Image.new("RGB", (100, 100), (250, 10, 10))
    image.paste(Image.new("RGB", (100, 25), (10, 10, 250)), (0, 75))
    descriptor = image_ops.rgb_descriptor(image)
    hist = np.array(descriptor.histogram)
    assert hist.shape == (spec.RGB_HIST_DIM,) and np.isfinite(hist).all()
    assert hist.sum() == pytest.approx(1.0, abs=1e-12)
    red_bin = (3 * 4 + 0) * 4 + 0
    blue_bin = (0 * 4 + 0) * 4 + 3
    assert hist[red_bin] == pytest.approx(0.75) and hist[blue_bin] == pytest.approx(0.25)
    assert descriptor.dominant[0]["bin"] == red_bin
    assert descriptor.dominant[0]["rgb"] == [250, 10, 10]
    assert image_ops.rgb_descriptor(image).histogram == descriptor.histogram  # deterministic
    # Values are uint8 pixels, not ImageNet-normalized tensors: a black crop is one bin.
    black = np.array(image_ops.rgb_descriptor(Image.new("RGB", (40, 40))).histogram)
    assert black[0] == 1.0


def test_canonical_loader_rejections():
    with pytest.raises(image_ops.ImageRejected) as error:
        image_ops.load_canonical(b"garbage")
    assert error.value.reason == spec.INVALID_IMAGE
    png = io.BytesIO()
    Image.new("RGB", (64, 64)).save(png, "PNG")
    with pytest.raises(image_ops.ImageRejected) as error:
        image_ops.load_canonical(png.getvalue())
    assert error.value.reason == spec.UNSUPPORTED_IMAGE
    loaded = image_ops.load_canonical(canonical(Image.new("RGB", (64, 64), "white")))
    with pytest.raises(image_ops.ImageRejected) as error:
        image_ops.apply_crop(loaded, spec.BBox(0, 0, 0.1, 0.1))
    assert error.value.reason == spec.INVALID_CROP


def test_user_text_normalization_keeps_mixed_breeds():
    assert normalize_text("  Немецкая   Овчарка + ТЕРЬЕР!! ") == "немецкая овчарка + терьер"
    assert normalize_text("Чёрный, с белым") == "черный с белым"
    assert normalize_text("   ") is None and normalize_text(None) is None


def test_backbone_specs_and_pipeline_ids_are_versioned():
    from app.cv.pipeline import resolve

    settings = Settings(_env_file=None, cv_mode="pretrained_features")
    pipeline = resolve(settings)
    assert pipeline.mode == "pretrained_features" and pipeline.bundle_id is None
    assert pipeline.embedding_space_id.startswith("efficientnet_b0/IMAGENET1K_V1/pp-")
    other = resolve(Settings(_env_file=None, cv_mode="pretrained_features", cv_backbone="resnet50"))
    assert other.embedding_space_id != pipeline.embedding_space_id
    assert spec.BACKBONES["efficientnet_b0"].dimension == 1280
    assert spec.BACKBONES["resnet50"].dimension == 2048
    gate_changed = resolve(
        Settings(_env_file=None, cv_mode="pretrained_features", cv_dog_score_threshold=0.6)
    )
    assert gate_changed.pipeline_id != pipeline.pipeline_id


def test_cv_settings_validation(tmp_path):
    from pydantic import ValidationError

    from app.cv.pipeline import PipelineConfigError, resolve

    with pytest.raises(ValidationError):
        Settings(_env_file=None, cv_mode="trained_heads")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cv_fusion="embedding-rgb-heads-v1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cv_fusion_rgb_weight=0.8, cv_fusion_heads_weight=0.3)
    # trained_heads with a missing bundle is an explicit error, never a silent fallback.
    with pytest.raises(PipelineConfigError):
        resolve(Settings(_env_file=None, cv_mode="trained_heads", cv_bundle_dir=tmp_path))
    with pytest.raises(PipelineConfigError):
        resolve(Settings(_env_file=None, cv_mode="mock"))


def test_retrieval_metric_definitions_differ():
    from app.ml import metrics

    # One query with 4 relevant gallery items; 2 of them are in the top-2 list.
    relevance = np.array([[True, True, True, True, False, False]])
    scores = np.array([[0.9, 0.8, 0.1, 0.05, 0.5, 0.4]])
    excluded = np.zeros_like(relevance)
    task = metrics.RetrievalTask(["q"], list("abcdef"), relevance, excluded, {})
    result = metrics.retrieval_metrics(task, scores, [2, 4])
    assert result["at_k"]["2"] == {"recall": 0.5, "hit_rate": 1.0, "precision": 1.0}
    assert result["at_k"]["4"]["recall"] == 0.5 and result["at_k"]["4"]["precision"] == 0.5
    # Excluded pairs (self / near-duplicates) are never ranked or counted as relevant.
    excluded = np.array([[True, False, False, False, False, False]])
    task = metrics.RetrievalTask(["q"], list("abcdef"), relevance, excluded, {})
    assert metrics.retrieval_metrics(task, scores, [1])["at_k"]["1"]["recall"] == 1 / 3
    # Queries without any relevant item are reported separately.
    task = metrics.RetrievalTask(
        ["q"], list("ab"), np.array([[False, False]]), np.zeros((1, 2), bool), {}
    )
    result = metrics.retrieval_metrics(task, np.array([[1.0, 0.0]]), [1])
    assert result["queries_without_relevant"] == 1 and result["at_k"]["1"]["recall"] is None


def test_classification_and_histogram_metrics():
    from app.ml import metrics

    result = metrics.classification_metrics(
        np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), ["a", "b"]
    )
    assert result["accuracy"] == 0.75 and result["macro_recall"] == 0.75
    assert result["per_class_recall"] == {"a": 0.5, "b": 1.0}
    target = np.array([[0.5, 0.5, 0.0]])
    assert metrics.histogram_metrics(target, target)["kl_mean"] == pytest.approx(0.0)
    assert metrics.kl_divergence(target, np.array([[0.25, 0.25, 0.5]]))[0] == pytest.approx(
        math.log(2)
    )
