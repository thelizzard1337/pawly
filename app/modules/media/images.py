import hashlib
import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImage(ValueError):
    """Invalid, animated, or oversized decoded image."""


@dataclass(frozen=True)
class NormalizedImage:
    content: bytes
    sha256: str
    width: int
    height: int


def normalize_image(raw: bytes, max_pixels: int) -> NormalizedImage:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as original:
                if original.format not in ("JPEG", "PNG", "WEBP"):
                    raise InvalidImage("Unsupported image format")
                if getattr(original, "n_frames", 1) != 1:
                    raise InvalidImage("Animated images are unsupported")
                if original.width * original.height > max_pixels:
                    raise InvalidImage("Too many pixels")
                original.verify()
            with Image.open(io.BytesIO(raw)) as original:
                original.load()
                oriented = ImageOps.exif_transpose(original)
                # A fresh RGB image prevents metadata/EXIF/GPS from surviving re-encoding.
                cleaned = Image.new("RGB", oriented.size, "white")
                if "A" in oriented.getbands():
                    rgba = oriented.convert("RGBA")
                    cleaned.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    cleaned.paste(oriented.convert("RGB"))
                result = io.BytesIO()
                cleaned.save(result, format="JPEG", quality=90, optimize=True)
                content = result.getvalue()
                return NormalizedImage(
                    content, hashlib.sha256(content).hexdigest(), cleaned.width, cleaned.height
                )
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise InvalidImage("Invalid or unsupported image") from exc
