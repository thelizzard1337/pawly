"""Versioned, torch-free contracts shared by the API, the CV worker and the training code."""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal

CVMode = Literal["mock", "pretrained_features", "trained_heads"]
BackboneName = Literal["efficientnet_b0", "resnet50"]

# Domain rejection reasons. They are outcomes of a valid request, not worker failures.
NO_DOG = "no_dog"
MULTIPLE_DOGS = "multiple_dogs"
INVALID_CROP = "invalid_crop"
INVALID_IMAGE = "invalid_image"
UNSUPPORTED_IMAGE = "unsupported_image"
REJECTION_REASONS = (NO_DOG, MULTIPLE_DOGS, INVALID_CROP, INVALID_IMAGE, UNSUPPORTED_IMAGE)

# Crop rule crop-v1:
#   bbox = (x_min, y_min, x_max, y_max), normalized to [0, 1] relative to the canonical image,
#   i.e. after EXIF orientation was applied by the upload pipeline.
#   Pixel edges: value * size is rounded to 6 decimals (removes float noise such as
#   0.3 * 100 = 30.000000000000004), then left/top use floor and right/bottom use ceil, so the
#   pixel box always contains the requested region. The result is clamped to the image.
#   Each side must be at least MIN_CROP_SIDE_PX pixels. bbox=None means the full image.
CROP_RULE = "crop-v1"
MIN_CROP_SIDE_PX = 32
BBOX_DECIMALS = 6

# RGB descriptor: joint RGB histogram with 4 bins per channel (64 bins, bin = value >> 6),
# computed on the uint8 sRGB pixels of the user crop before any resize/normalization/padding,
# normalized to sum 1. Up to DOMINANT_COLORS bins are also stored with their mean RGB.
RGB_DESCRIPTOR_VERSION = "rgb-hist-4x4x4-v1"
RGB_BINS_PER_CHANNEL = 4
RGB_HIST_DIM = RGB_BINS_PER_CHANNEL**3
DOMINANT_COLORS = 3

GATE_ID = "ssdlite320_mobilenet_v3_large.COCO_V1"


class InvalidCrop(ValueError):
    """The requested crop is malformed or too small for the image."""


@dataclass(frozen=True)
class BBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
            raise InvalidCrop("bbox values must be finite numbers")
        if not all(0.0 <= v <= 1.0 for v in values):
            raise InvalidCrop("bbox values must be in [0, 1]")
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise InvalidCrop("bbox must have x_min < x_max and y_min < y_max")

    def as_list(self) -> list[float]:
        return [round(float(v), BBOX_DECIMALS) for v in self.astuple()]

    def astuple(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @classmethod
    def from_values(cls, values) -> "BBox":
        try:
            items = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise InvalidCrop("bbox must contain four numbers") from exc
        if len(items) != 4:
            raise InvalidCrop("bbox must contain four numbers")
        return cls(*items)

    @classmethod
    def parse(cls, raw: str) -> "BBox":
        """Parse ``"x_min,y_min,x_max,y_max"`` (multipart form value)."""
        if len(raw) > 200:
            raise InvalidCrop("bbox is too long")
        return cls.from_values(part.strip() for part in raw.split(","))


@dataclass(frozen=True)
class CropSpec:
    rule: str
    bbox: list[float] | None
    box: tuple[int, int, int, int]  # left, top, right, bottom in canonical pixels
    image_size: tuple[int, int]

    @property
    def hash(self) -> str:
        payload = {
            "rule": self.rule,
            "bbox": self.bbox,
            "box": list(self.box),
            "size": list(self.image_size),
        }
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _edge(value: float, size: int, ceil: bool) -> int:
    scaled = round(value * size, BBOX_DECIMALS)
    edge = math.ceil(scaled) if ceil else math.floor(scaled)
    return min(max(edge, 0), size)


def crop_spec(bbox: BBox | None, width: int, height: int) -> CropSpec:
    if width <= 0 or height <= 0:
        raise InvalidCrop("image has no pixels")
    if bbox is None:
        box = (0, 0, width, height)
        normalized = None
    else:
        box = (
            _edge(bbox.x_min, width, False),
            _edge(bbox.y_min, height, False),
            _edge(bbox.x_max, width, True),
            _edge(bbox.y_max, height, True),
        )
        normalized = bbox.as_list()
    if box[2] - box[0] < MIN_CROP_SIDE_PX or box[3] - box[1] < MIN_CROP_SIDE_PX:
        raise InvalidCrop(f"crop must be at least {MIN_CROP_SIDE_PX}x{MIN_CROP_SIDE_PX} pixels")
    return CropSpec(CROP_RULE, normalized, box, (width, height))


def bbox_or_none(values) -> BBox | None:
    return None if values is None else BBox.from_values(values)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class BackboneSpec:
    """Static description of a torchvision backbone; verified against torchvision at load."""

    name: str
    weights_id: str
    weights_file: str
    dimension: int
    # Deterministic eval transforms of these weights (torchvision ImageClassification).
    resize_size: int
    crop_size: int
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def preprocessing(self) -> dict:
        return {
            "input": "canonical sRGB JPEG -> user crop (crop-v1) -> PIL RGB",
            "resize_shorter_side": self.resize_size,
            "center_crop": self.crop_size,
            "interpolation": self.interpolation,
            "antialias": True,
            "to_tensor": "float32 in [0, 1]",
            "normalize_mean": list(self.mean),
            "normalize_std": list(self.std),
        }

    @property
    def preprocessing_id(self) -> str:
        digest = hashlib.sha256(canonical_json(self.preprocessing).encode()).hexdigest()
        return f"pp-{digest[:12]}"

    @property
    def embedding_space_id(self) -> str:
        # Pooled features before the classifier, L2-normalized.
        return f"{self.name}/{self.weights_id}/{self.preprocessing_id}/gap-l2"


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES: dict[str, BackboneSpec] = {
    "efficientnet_b0": BackboneSpec(
        name="efficientnet_b0",
        weights_id="IMAGENET1K_V1",
        weights_file="efficientnet_b0_rwightman-7f5810bc.pth",
        dimension=1280,
        resize_size=256,
        crop_size=224,
        interpolation="bicubic",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
    "resnet50": BackboneSpec(
        name="resnet50",
        weights_id="IMAGENET1K_V2",
        weights_file="resnet50-11ad3fa6.pth",
        dimension=2048,
        resize_size=232,
        crop_size=224,
        interpolation="bilinear",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
}


def gate_id(threshold: float) -> str:
    return f"{GATE_ID}@{threshold:.2f}"
