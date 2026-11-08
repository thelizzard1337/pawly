"""Frozen torchvision backbones and the single preprocessing implementation (train == runtime)."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import InterpolationMode

from app.cv import spec
from app.cv.weights import REGISTRY, verify

_WEIGHT_ENUMS = {
    "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
}


class PreprocessingMismatch(RuntimeError):
    pass


def weights_enum(name: str):
    return _WEIGHT_ENUMS[name]


def check_preprocessing(backbone: spec.BackboneSpec) -> None:
    """Fail if torchvision's transforms differ from the versioned static specification."""
    transform = weights_enum(backbone.name).transforms()
    actual = {
        "resize": list(transform.resize_size),
        "crop": list(transform.crop_size),
        "interpolation": transform.interpolation,
        "mean": [round(v, 6) for v in transform.mean],
        "std": [round(v, 6) for v in transform.std],
        "antialias": transform.antialias,
    }
    expected = {
        "resize": [backbone.resize_size],
        "crop": [backbone.crop_size],
        "interpolation": InterpolationMode(backbone.interpolation),
        "mean": [round(v, 6) for v in backbone.mean],
        "std": [round(v, 6) for v in backbone.std],
        "antialias": True,
    }
    if actual != expected:
        raise PreprocessingMismatch(f"{backbone.name}: {actual} != {expected}")


def make_preprocess(backbone: spec.BackboneSpec):
    """PIL RGB crop -> normalized float tensor [3, crop, crop]. Used for training and inference.

    torchvision's eval transform resizes the shorter side and then center-crops, so pixels at
    the long edges of a non-square crop are not seen by the backbone. The RGB descriptor is
    computed separately on the full user crop.
    """
    check_preprocessing(backbone)
    transform = weights_enum(backbone.name).transforms()

    def preprocess(image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            raise ValueError("preprocess expects an RGB image")
        return transform(image)

    return preprocess


class FrozenEncoder(nn.Module):
    """Pooled features before the classifier (EfficientNet avgpool / ResNet avgpool)."""

    def __init__(self, name: str, trunk: nn.Module, dimension: int):
        super().__init__()
        self.name, self.trunk, self.dimension = name, trunk, dimension

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.trunk(images), 1)


def _architecture(name: str) -> tuple[nn.Module, nn.Module]:
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        trunk = nn.Sequential(model.features, model.avgpool)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        trunk = nn.Sequential(*list(model.children())[:-1])
    else:
        raise ValueError(f"unknown backbone {name}")
    return model, trunk


def build_encoder(name: str, weights_dir: Path | None) -> FrozenEncoder:
    """Real pretrained encoder from pinned local weights; weights_dir=None only for shape tests."""
    backbone = spec.BACKBONES[name]
    model, trunk = _architecture(name)
    if weights_dir is not None:
        path = verify(name, weights_dir)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)  # strict: every tensor must match the architecture
    encoder = FrozenEncoder(name, trunk, backbone.dimension)
    freeze(encoder)
    return encoder


def freeze(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    norms = features.norm(dim=1, keepdim=True)
    if not torch.isfinite(features).all() or (norms <= 1e-12).any():
        raise ValueError("embedding is not finite or has zero norm")
    return features / norms


assert set(REGISTRY) >= set(spec.BACKBONES)
