"""Pinned official torchvision weights. Standard library only (used in the Docker build).

Usage:
    python -m app.cv.weights download --dir /opt/pawly/weights efficientnet_b0 ssdlite
    python -m app.cv.weights verify --dir /opt/pawly/weights efficientnet_b0 ssdlite
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightsFile:
    key: str
    torchvision_enum: str
    filename: str
    url: str
    sha256: str
    license_note: str


_BASE = "https://download.pytorch.org/models/"

# SHA-256 values were computed from the official download.pytorch.org files on 2026-11-08; the
# 8-hex prefix in each filename is the hash prefix torchvision itself uses for verification.
REGISTRY: dict[str, WeightsFile] = {
    "efficientnet_b0": WeightsFile(
        "efficientnet_b0",
        "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "efficientnet_b0_rwightman-7f5810bc.pth",
        _BASE + "efficientnet_b0_rwightman-7f5810bc.pth",
        "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934",
        "torchvision pretrained weights (ImageNet-1k); see torchvision model docs",
    ),
    "resnet50": WeightsFile(
        "resnet50",
        "ResNet50_Weights.IMAGENET1K_V2",
        "resnet50-11ad3fa6.pth",
        _BASE + "resnet50-11ad3fa6.pth",
        "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca",
        "torchvision pretrained weights (ImageNet-1k); see torchvision model docs",
    ),
    "ssdlite": WeightsFile(
        "ssdlite",
        "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
        "ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
        _BASE + "ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
        "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2",
        "torchvision pretrained weights (COCO 2017); see torchvision model docs",
    ),
}


class WeightsError(RuntimeError):
    """Missing or corrupted weights file; never silently replaced."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_for(key: str, directory: Path) -> Path:
    return directory / REGISTRY[key].filename


def verify(key: str, directory: Path) -> Path:
    entry = REGISTRY[key]
    path = path_for(key, directory)
    if not path.is_file():
        raise WeightsError(f"weights_missing:{entry.filename}")
    actual = sha256_file(path)
    if actual != entry.sha256:
        raise WeightsError(f"weights_checksum_mismatch:{entry.filename}")
    return path


def download(key: str, directory: Path) -> Path:
    entry = REGISTRY[key]
    directory.mkdir(parents=True, exist_ok=True)
    target = path_for(key, directory)
    if target.is_file() and sha256_file(target) == entry.sha256:
        return target
    with tempfile.NamedTemporaryFile(dir=directory, delete=False, suffix=".part") as handle:
        temporary = Path(handle.name)
        digest = hashlib.sha256()
        with urllib.request.urlopen(entry.url, timeout=60) as response:  # noqa: S310 pinned URL
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
                handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    if digest.hexdigest() != entry.sha256:
        temporary.unlink(missing_ok=True)
        raise WeightsError(f"weights_checksum_mismatch:{entry.filename}")
    os.replace(temporary, target)
    os.chmod(target, 0o644)  # read-only for the non-root runtime user
    return target


def write_manifest(directory: Path, keys: list[str]) -> None:
    manifest = {key: {**asdict(REGISTRY[key]), "verified": True} for key in keys}
    (directory / "weights-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "verify"])
    parser.add_argument("keys", nargs="+", choices=sorted(REGISTRY))
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for key in args.keys:
        path = download(key, args.dir) if args.action == "download" else verify(key, args.dir)
        print(f"{args.action} ok: {path.name} sha256={REGISTRY[key].sha256}")
    if args.action == "download":
        write_manifest(args.dir, args.keys)


if __name__ == "__main__":
    try:
        main()
    except WeightsError as exc:
        sys.exit(str(exc))
