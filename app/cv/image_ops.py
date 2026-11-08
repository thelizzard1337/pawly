"""Canonical image decoding, user crop and the RGB descriptor (Pillow + numpy, no torch)."""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.cv import spec


class ImageRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def load_canonical(raw: bytes) -> Image.Image:
    """Decode a canonical upload (JPEG written by normalize_image, already EXIF-oriented).

    The upload pipeline drops embedded ICC profiles without colour conversion, so pixel values
    are interpreted as sRGB. Other formats are not produced by the upload pipeline.
    """
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG":
                raise ImageRejected(spec.UNSUPPORTED_IMAGE)
            image.load()
            if image.mode != "RGB":
                raise ImageRejected(spec.UNSUPPORTED_IMAGE)
            return image.copy()
    except ImageRejected:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageRejected(spec.INVALID_IMAGE) from exc


def apply_crop(image: Image.Image, bbox: spec.BBox | None) -> tuple[Image.Image, spec.CropSpec]:
    try:
        crop = spec.crop_spec(bbox, image.width, image.height)
    except spec.InvalidCrop as exc:
        raise ImageRejected(spec.INVALID_CROP) from exc
    return image.crop(crop.box), crop


@dataclass(frozen=True)
class RGBDescriptor:
    histogram: list[float]
    dominant: list[dict]
    version: str = spec.RGB_DESCRIPTOR_VERSION


def rgb_descriptor(crop: Image.Image) -> RGBDescriptor:
    """Joint RGB histogram of the user crop (uint8 sRGB pixels, before any model transform).

    Without a segmentation mask this describes the whole crop, background included; it is not a
    verified coat colour.
    """
    if crop.mode != "RGB":
        raise ValueError("RGB descriptor expects an RGB image")
    pixels = np.asarray(crop, dtype=np.uint8).reshape(-1, 3)
    if pixels.size == 0:
        raise ImageRejected(spec.INVALID_CROP)
    shift = 8 - int(np.log2(spec.RGB_BINS_PER_CHANNEL))
    bins = spec.RGB_BINS_PER_CHANNEL
    channel_bins = (pixels >> shift).astype(np.int64)
    index = (channel_bins[:, 0] * bins + channel_bins[:, 1]) * bins + channel_bins[:, 2]
    counts = np.bincount(index, minlength=spec.RGB_HIST_DIM).astype(np.float64)
    histogram = counts / counts.sum()
    # Dominant bins: highest mass first, ties broken by bin index; mean RGB of the member pixels.
    order = sorted(range(spec.RGB_HIST_DIM), key=lambda b: (-counts[b], b))
    dominant = []
    for b in order[: spec.DOMINANT_COLORS]:
        if counts[b] == 0:
            break
        mean = pixels[index == b].mean(axis=0)
        dominant.append(
            {
                "bin": int(b),
                "share": round(float(histogram[b]), 6),
                "rgb": [int(round(float(v))) for v in mean],
            }
        )
    return RGBDescriptor([float(v) for v in histogram], dominant)
