"""Model mechanics with random-initialised architectures (no pretrained download needed).

These tests check shapes, freezing and bundle validation. They prove nothing about ML quality.
"""

import io
import json
import math
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from app.cv import spec  # noqa: E402
from app.cv.backbones import (  # noqa: E402
    build_encoder,
    check_preprocessing,
    l2_normalize,
    make_preprocess,
)
from app.cv.bundle import BundleError, export_bundle, load_bundle  # noqa: E402
from app.cv.heads import (  # noqa: E402
    UNKNOWN_LABEL,
    HeadConfig,
    TwoHeadModel,
    TwoHeads,
    breed_loss,
    color_loss,
    make_optimizer,
    train_step,
)
from app.modules.media.images import normalize_image  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _calibrated(name):
    """Random weights give ~0 features with default BN statistics; calibrate them once."""
    encoder = build_encoder(name, None)
    encoder.train()
    with torch.no_grad():
        for _ in range(3):
            encoder(torch.rand(4, 3, 224, 224))
    encoder.eval()
    return encoder


@pytest.fixture(scope="module")
def encoders():
    torch.manual_seed(0)
    return {name: _calibrated(name) for name in ("efficientnet_b0", "resnet50")}


@pytest.mark.parametrize("name", ["efficientnet_b0", "resnet50"])
def test_embedding_shapes_and_l2(encoders, name):
    encoder = encoders[name]
    images = torch.rand(2, 3, 224, 224)
    with torch.inference_mode():
        features = encoder(images)
    assert features.shape == (2, spec.BACKBONES[name].dimension)
    normalized = l2_normalize(features)
    assert torch.allclose(normalized.norm(dim=1), torch.ones(2), atol=1e-5)
    with pytest.raises(ValueError):
        l2_normalize(torch.zeros(1, 4))
    with pytest.raises(ValueError):
        l2_normalize(torch.tensor([[math.nan, 1.0]]))


@pytest.mark.parametrize("name", ["efficientnet_b0", "resnet50"])
def test_static_preprocessing_matches_torchvision(name):
    check_preprocessing(spec.BACKBONES[name])  # raises on any drift
    preprocess = make_preprocess(spec.BACKBONES[name])
    # Center crop: a wide crop loses its long edges; the documented behaviour is tested here.
    tensor = preprocess(Image.new("RGB", (600, 300), "white"))
    assert tensor.shape == (3, 224, 224)


def test_head_only_step_keeps_backbone_parameters_and_buffers(encoders):
    encoder = encoders["efficientnet_b0"]
    heads = TwoHeads(HeadConfig(1280, 3, hidden=16, dropout=0.1))
    model = TwoHeadModel(encoder, heads)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}  # params + BN buffers
    optimizer = make_optimizer(heads, 1e-2, 0.0)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in heads.parameters()}
    assert all(not p.requires_grad for p in encoder.parameters())
    head_before = {k: v.clone() for k, v in heads.state_dict().items()}
    labels = torch.tensor([0, 1, UNKNOWN_LABEL, 2])
    targets = torch.softmax(torch.rand(4, 64), dim=1)
    for _ in range(2):
        train_step(model, optimizer, torch.rand(4, 3, 224, 224), labels, targets)
    assert model.training and heads.training and not encoder.training
    for key, value in encoder.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert any(not torch.equal(v, head_before[k]) for k, v in heads.state_dict().items())


def test_losses():
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, UNKNOWN_LABEL, 2, UNKNOWN_LABEL])
    expected = torch.nn.functional.cross_entropy(logits[[0, 2]], labels[[0, 2]])
    assert torch.allclose(breed_loss(logits, labels), expected)
    assert breed_loss(logits, torch.full((4,), UNKNOWN_LABEL)).item() == 0.0
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    assert color_loss(torch.log(target.clamp_min(1e-30)), target).item() == pytest.approx(
        0, abs=1e-6
    )
    uniform = torch.zeros(1, 4)
    # KL(t || uniform) = sum t log(t / 0.25) = log 2 for t = [0.5, 0.5, 0, 0].
    assert color_loss(uniform, target).item() == pytest.approx(math.log(2), rel=1e-6)
    # The predicted histogram is a distribution: components sum to 1.
    assert torch.softmax(torch.randn(3, 64), 1).sum(1) == pytest.approx(torch.ones(3))


def _heads(classes=3):
    torch.manual_seed(1)
    return TwoHeads(HeadConfig(1280, classes))


def _export(tmp_path, name="bundle-a", classes=("beagle", "pug", "boxer")):
    return export_bundle(
        tmp_path / name,
        _heads(len(classes)),
        "efficientnet_b0",
        list(classes),
        "unit-test",
        0.4,
        training={"run_id": "t"},
        metrics={},
    )


def test_bundle_roundtrip(tmp_path):
    heads = _heads()
    path = export_bundle(
        tmp_path / "b1", heads, "efficientnet_b0", ["a", "b", "c"], "unit", 0.3, {}, {}
    )
    bundle = load_bundle(path, "efficientnet_b0")
    manifest = json.loads((path / "manifest.json").read_text())
    assert bundle.bundle_id == "b1+" + manifest["heads"]["sha256"][:12]
    assert bundle.classes == ["a", "b", "c"]
    other = export_bundle(
        tmp_path / "b2", _heads(), "efficientnet_b0", ["a", "b", "c"], "u", 0.3, {}, {}
    )
    retrained = TwoHeads(HeadConfig(1280, 3))
    third = export_bundle(
        tmp_path / "b3", retrained, "efficientnet_b0", ["a", "b", "c"], "u", 0.3, {}, {}
    )
    # Same name prefix/weights -> same suffix; different weights -> different bundle_id.
    assert (
        load_bundle(other, "efficientnet_b0").bundle_id.split("+")[1]
        == bundle.bundle_id.split("+")[1]
    )
    assert (
        load_bundle(third, "efficientnet_b0").bundle_id.split("+")[1]
        != bundle.bundle_id.split("+")[1]
    )
    features = torch.rand(2, 1280)
    for a, b in zip(heads(features), bundle.heads(features), strict=True):
        assert torch.allclose(a, b)
    with pytest.raises(BundleError):
        export_bundle(tmp_path / "b1", heads, "efficientnet_b0", ["a", "b", "c"], "u", 0.3, {}, {})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m, p: m["taxonomy"]["classes"].append("extra"),
        lambda m, p: m.__setitem__("embedding_dim", 2048),
        lambda m, p: m.__setitem__("preprocessing_id", "pp-other"),
        lambda m, p: m["preprocessing"].__setitem__("center_crop", 256),
        lambda m, p: m.__setitem__("descriptor_version", "rgb-hist-8x8x8-v9"),
        lambda m, p: m.__setitem__("weights_sha256", "0" * 64),
        lambda m, p: m.__setitem__("schema", "other"),
        lambda m, p: m.__setitem__("bundle_id", "renamed-without-hash"),
        lambda m, p: (p / "heads.pt").write_bytes(b"corrupted"),
        lambda m, p: (p / "heads.pt").unlink(),
    ],
)
def test_incompatible_bundles_are_rejected(tmp_path, mutate):
    path = _export(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    mutate(manifest, path)
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError):
        load_bundle(path, "efficientnet_b0")


def test_bundle_for_other_backbone_is_rejected(tmp_path):
    path = _export(tmp_path)
    with pytest.raises(BundleError):
        load_bundle(path, "resnet50")
    with pytest.raises(BundleError):
        load_bundle(tmp_path / "missing", "efficientnet_b0")


class _AcceptGate:
    def check(self, crop):
        from app.cv.gate import GateResult

        return GateResult(True, None, 1, 1.0)


def _pipeline(mode="pretrained_features", bundle_id=None):
    from app.cv.pipeline import PipelineSpec

    return PipelineSpec(mode, spec.BACKBONES["efficientnet_b0"], 0.5, bundle_id, "embedding-v1")


def _canonical_fixture():
    return normalize_image((FIXTURES / "dog_beagle_pd.jpg").read_bytes(), 20_000_000)


def test_pretrained_mode_never_returns_head_predictions(encoders):
    from app.cv.extractor import FeatureExtractor

    extractor = FeatureExtractor(_pipeline(), encoders["efficientnet_b0"], _AcceptGate(), None)
    result = extractor.extract(_canonical_fixture().content, None)
    assert result.rejected is None and len(result.embedding) == 1280
    assert result.breed_prediction is None and result.color_prediction is None
    assert sum(result.rgb_histogram) == pytest.approx(1.0)
    with pytest.raises(ValueError):  # trained_heads without a bundle cannot be constructed
        FeatureExtractor(
            _pipeline("trained_heads", "x"), encoders["efficientnet_b0"], _AcceptGate(), None
        )


def test_trained_heads_mode_uses_only_the_loaded_bundle(tmp_path, encoders):
    from app.cv.extractor import FeatureExtractor

    bundle = load_bundle(_export(tmp_path), "efficientnet_b0")
    extractor = FeatureExtractor(
        _pipeline("trained_heads", bundle.bundle_id),
        encoders["efficientnet_b0"],
        _AcceptGate(),
        bundle,
    )
    result = extractor.extract(_canonical_fixture().content, spec.BBox(0.1, 0.1, 0.9, 0.9))
    prediction = result.breed_prediction
    assert {item["class"] for item in prediction["top"]} <= {"beagle", "pug", "boxer"}
    assert sum(prediction["distribution"]) == pytest.approx(1.0, abs=1e-4)
    assert len(result.color_prediction) == 64
    assert sum(result.color_prediction) == pytest.approx(1.0, abs=1e-4)


def test_train_and_inference_preprocessing_parity(tmp_path, encoders):
    """The dataset loader and the runtime extractor produce identical tensors and histograms."""
    from app.cv import image_ops
    from app.ml.features import load_sample

    canonical = _canonical_fixture()
    raw = io.BytesIO()
    Image.open(FIXTURES / "dog_beagle_pd.jpg").save(raw, "JPEG")
    (tmp_path / "img.jpg").write_bytes((FIXTURES / "dog_beagle_pd.jpg").read_bytes())
    row = {
        "sample_id": "s",
        "rel_path": "img.jpg",
        "canonical_sha256": canonical.sha256,
        "bbox": [0.1, 0.2, 0.9, 0.95],
    }
    preprocess = make_preprocess(spec.BACKBONES["efficientnet_b0"])
    train_tensor, train_hist = load_sample(row, tmp_path, preprocess)
    image = image_ops.load_canonical(canonical.content)
    crop, _ = image_ops.apply_crop(image, spec.BBox(0.1, 0.2, 0.9, 0.95))
    assert torch.equal(train_tensor, preprocess(crop))
    assert train_hist.tolist() == pytest.approx(image_ops.rgb_descriptor(crop).histogram)
