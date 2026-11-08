"""Dataset adapters and manifests.

Oxford-IIIT Pet (default, CC BY-SA 4.0): dog subset, breed labels from annotations/list.txt.
Stanford Dogs (optional): ImageNet-derived images; requires an explicit terms acknowledgement.

Every image goes through the production upload normalisation (``normalize_image``) so training
and runtime see the same canonical sRGB JPEG.
"""

import hashlib
import io
import json
import os
import random
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.cv import spec
from app.modules.media.images import InvalidImage, normalize_image

MAX_PIXELS = 20_000_000


@dataclass(frozen=True)
class Archive:
    name: str
    url: str
    sha256: str | None  # None: not pinned, computed and recorded at preparation time
    md5: str | None = None


OXFORD = {
    "name": "oxford-iiit-pet",
    "homepage": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
    "license": "CC BY-SA 4.0 (dataset page: available for commercial/research purposes; "
    "copyright of images remains with the original owners)",
    "archives": [
        # SHA-256 computed 2026-11-08; MD5 equals the values used by torchvision's
        # OxfordIIITPet dataset class (independent cross-check of the files).
        Archive(
            "images.tar.gz",
            "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
            "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d",
            "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        ),
        Archive(
            "annotations.tar.gz",
            "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
            "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
            "95a8c909bbe2e81eed6a22bccdf3f68f",
        ),
    ],
    "crop_policy": "full-image: Oxford XML boxes are HEAD boxes and are not used as dog crops",
    "identity": "unavailable: no individual-animal IDs; near-duplicate grouping does not prove "
    "absence of identity leakage between splits",
}

STANFORD = {
    "name": "stanford-dogs",
    "homepage": "http://vision.stanford.edu/aditya86/ImageNetDogs/main.html",
    "license": "Images from ImageNet; ImageNet terms restrict use to non-commercial research and "
    "educational purposes. Requires --accept-terms imagenet-noncommercial-research.",
    "terms_flag": "imagenet-noncommercial-research",
    "archives": [
        Archive("images.tar", "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", None),
        Archive(
            "annotation.tar",
            "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar",
            None,
        ),
    ],
    "crop_policy": "full-dog bbox from the dataset annotation (single-object images only)",
    "identity": "unavailable: no individual-animal IDs",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_archive(archive: Archive, raw_dir: Path, local_dir: Path | None) -> dict:
    """Download (or copy from --archives-dir) and verify; repeat-safe."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / archive.name
    if not (target.is_file() and (archive.sha256 is None or sha256_file(target) == archive.sha256)):
        temporary = target.with_suffix(target.suffix + ".part")
        if local_dir is not None and (local_dir / archive.name).is_file():
            shutil.copyfile(local_dir / archive.name, temporary)
        else:
            with urllib.request.urlopen(archive.url, timeout=120) as response:  # noqa: S310
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, 1024 * 1024)
        os.replace(temporary, target)
    actual = sha256_file(target)
    if archive.sha256 is not None and actual != archive.sha256:
        target.unlink()
        raise SystemExit(f"Checksum mismatch for {archive.name}; file removed")
    return {
        "name": archive.name,
        "url": archive.url,
        "sha256": actual,
        "pinned": archive.sha256 is not None,
    }


def safe_extract(archive: Path, destination: Path, allowed_prefixes: tuple[str, ...]) -> int:
    """Extract regular files under allowed prefixes only; rejects traversal and links."""
    destination = destination.resolve()
    count = 0
    with tarfile.open(archive) as tar:
        for member in tar:
            name = member.name.lstrip("./")
            if not member.isfile() or not name.startswith(allowed_prefixes):
                continue
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise SystemExit(f"Unsafe path in archive: {member.name}")
            if target.exists() and target.stat().st_size == member.size:
                count += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            count += 1
    return count


def dhash(image: Image.Image, vertical: bool = False) -> int:
    """64-bit difference hash (horizontal gradients, or vertical with ``vertical=True``)."""
    gray = image.convert("L")
    if vertical:
        gray = gray.transpose(Image.Transpose.TRANSPOSE)
    small = gray.resize((9, 8), Image.Resampling.BILINEAR)
    pixels = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return bits


def canonicalize(path_str: str):
    """Worker-process task: production normalisation + hashes."""
    path = Path(path_str)
    raw = path.read_bytes()
    try:
        normalized = normalize_image(raw, MAX_PIXELS)
    except InvalidImage:
        return path_str, None
    with Image.open(io.BytesIO(normalized.content)) as image:
        horizontal, vertical = dhash(image), dhash(image, vertical=True)
    return path_str, {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_sha256": normalized.sha256,
        "width": normalized.width,
        "height": normalized.height,
        "dhash": f"{horizontal:016x}",
        "dhash_v": f"{vertical:016x}",
    }


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_groups(
    horizontal: list[int], vertical: list[int], max_distance: int
) -> list[int]:
    """Union pairs whose horizontal AND vertical dHash are within ``max_distance`` bits.

    Requiring both hashes avoids chaining unrelated low-texture images into one large group
    (observed with a single horizontal dHash on Oxford-IIIT Pet).
    """
    import numpy as np

    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def distances(values, i):
        xor = np.bitwise_xor(values[i + 1 :], values[i])
        return popcount[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1, dtype=np.int64)

    h = np.array(horizontal, dtype=np.uint64)
    v = np.array(vertical, dtype=np.uint64)
    uf = UnionFind(len(h))
    for i in range(len(h)):
        close = (distances(h, i) <= max_distance) & (distances(v, i) <= max_distance)
        for j in np.nonzero(close)[0]:
            uf.union(i, i + 1 + int(j))
    return [uf.find(i) for i in range(len(h))]


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def parse_oxford(extracted: Path) -> list[dict]:
    official = {}
    for split_name, file_name in (("trainval", "trainval.txt"), ("test", "test.txt")):
        for line in (extracted / "annotations" / file_name).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                official[line.split()[0]] = split_name
    samples = []
    for line in (extracted / "annotations" / "list.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        image_id, _class_id, species, _breed_id = line.split()
        breed = image_id.rsplit("_", 1)[0].lower()
        samples.append(
            {
                "sample_id": f"oxford/{image_id}",
                "rel_path": f"images/{image_id}.jpg",
                "species": "dog" if species == "2" else "cat",
                "breed": breed,
                "official_split": official.get(image_id),
                "bbox": None,
            }
        )
    return samples


def parse_stanford(extracted: Path) -> list[dict]:
    samples = []
    for xml_path in sorted((extracted / "Annotation").glob("*/*")):
        root = ET.parse(xml_path).getroot()
        objects = root.findall("object")
        if len(objects) != 1:
            continue  # production gate rejects multiple dogs; keep single-dog images only
        size = root.find("size")
        width, height = int(size.findtext("width")), int(size.findtext("height"))
        box = objects[0].find("bndbox")
        bbox = [
            int(box.findtext("xmin")) / width,
            int(box.findtext("ymin")) / height,
            int(box.findtext("xmax")) / width,
            int(box.findtext("ymax")) / height,
        ]
        bbox = [min(max(round(v, 6), 0.0), 1.0) for v in bbox]
        synset_breed = xml_path.parent.name.split("-", 1)[1].lower()
        samples.append(
            {
                "sample_id": f"stanford/{xml_path.name}",
                "rel_path": f"Images/{xml_path.parent.name}/{xml_path.name}.jpg",
                "species": "dog",
                "breed": synset_breed,
                "official_split": None,
                "bbox": bbox,
            }
        )
    return samples


def assign_splits(samples: list[dict], seed: int, holdout_breeds: int, val_fraction: float):
    """Group-aware splits; held-out breeds simulate unknown breeds (never in training)."""
    dog_breeds = sorted({s["breed"] for s in samples if s["species"] == "dog"})
    rng = random.Random(seed)
    held = sorted(rng.sample(dog_breeds, holdout_breeds)) if holdout_breeds else []
    val_unknown, test_unknown = set(held[: len(held) // 2]), set(held[len(held) // 2 :])
    groups = defaultdict(list)
    for s in samples:
        groups[s["group_id"]].append(s)
    for members in groups.values():
        dogs = [m for m in members if m["species"] == "dog"]
        for m in members:
            if m["species"] != "dog":
                m["split"], m["label_status"] = "gate_negative", "non_dog"
        if not dogs:
            continue
        # A near-duplicate group never straddles splits; any official-test member -> test.
        breeds = {m["breed"] for m in dogs}
        # Test-unknown breeds take precedence so none of them can reach validation (threshold
        # selection) through a shared near-duplicate group.
        if breeds & test_unknown:
            split = "test"
        elif breeds & val_unknown:
            split = "val"
        elif any(m["official_split"] == "test" for m in dogs):
            split = "test"
        elif any(m["official_split"] is None for m in dogs):
            fraction = stable_fraction(dogs[0]["group_id"], seed)
            split = (
                "test" if fraction < 0.2 else "val" if fraction < 0.2 + val_fraction else "train"
            )
        else:
            split = "val" if stable_fraction(dogs[0]["group_id"], seed) < val_fraction else "train"
        for m in dogs:
            m["split"] = split
            unknown = m["breed"] in val_unknown or m["breed"] in test_unknown
            m["label_status"] = "heldout_unknown" if unknown else "known"
    known = sorted(set(dog_breeds) - set(held))
    return known, sorted(val_unknown), sorted(test_unknown)


def prepare(
    dataset: str,
    root: Path,
    archives_dir: Path | None,
    seed: int,
    holdout_breeds: int,
    val_fraction: float,
    near_dup_distance: int,
    workers: int,
    accept_terms: str | None,
):
    info = OXFORD if dataset == "oxford-pets" else STANFORD
    if dataset == "stanford-dogs" and accept_terms != STANFORD["terms_flag"]:
        raise SystemExit(STANFORD["license"])
    base = root / "datasets" / dataset
    raw_dir, extracted = base / "raw", base / "extracted"
    print(f"License/terms: {info['license']}")
    archives = [fetch_archive(a, raw_dir, archives_dir) for a in info["archives"]]
    if dataset == "oxford-pets":
        safe_extract(raw_dir / "annotations.tar.gz", extracted, ("annotations/",))
        safe_extract(raw_dir / "images.tar.gz", extracted, ("images/",))
        samples = parse_oxford(extracted)
    else:
        safe_extract(raw_dir / "annotation.tar", extracted, ("Annotation/",))
        safe_extract(raw_dir / "images.tar", extracted, ("Images/",))
        samples = parse_stanford(extracted)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = dict(
            pool.map(canonicalize, [str(extracted / s["rel_path"]) for s in samples], chunksize=32)
        )
    usable, invalid = [], []
    for s in samples:
        meta = results.get(str(extracted / s["rel_path"]))
        (usable if meta else invalid).append(s)
        if meta:
            s.update(meta)
    # Exact duplicates (same canonical bytes): keep the first by sample_id.
    seen, unique, exact_duplicates = {}, [], []
    for s in sorted(usable, key=lambda s: s["sample_id"]):
        if s["canonical_sha256"] in seen:
            exact_duplicates.append(
                {"sample_id": s["sample_id"], "duplicate_of": seen[s["canonical_sha256"]]}
            )
            continue
        seen[s["canonical_sha256"]] = s["sample_id"]
        unique.append(s)
    roots = near_duplicate_groups(
        [int(s["dhash"], 16) for s in unique],
        [int(s["dhash_v"], 16) for s in unique],
        near_dup_distance,
    )
    for s, r in zip(unique, roots, strict=True):
        s["group_id"] = unique[r]["sample_id"]
    known, val_unknown, test_unknown = assign_splits(unique, seed, holdout_breeds, val_fraction)
    for s in unique:
        s.update(
            {
                "source": info["name"],
                "license": info["license"].split(" (")[0],
                "class_name": s["breed"] if s.get("label_status") == "known" else None,
                "crop_policy": info["crop_policy"],
                "pseudo_target_version": spec.RGB_DESCRIPTOR_VERSION,
            }
        )
    manifest_path = base / "manifest.jsonl"
    lines = [spec.canonical_json(s) for s in unique]
    manifest_path.write_text("\n".join(lines) + "\n")
    manifest_sha = sha256_file(manifest_path)
    dog_groups = defaultdict(set)
    for s in unique:
        if s["species"] == "dog":
            dog_groups[s["group_id"]].add(s["split"])
    if any(len(v) > 1 for v in dog_groups.values()):
        raise RuntimeError("a near-duplicate dog group spans splits")
    counts = defaultdict(int)
    for s in unique:
        counts[f"{s['split']}:{s['label_status']}"] += 1
    summary = {
        "dataset": info["name"],
        "homepage": info["homepage"],
        "license": info["license"],
        "archives": archives,
        "seed": seed,
        "manifest": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "taxonomy": known,
        "heldout_unknown_breeds": {"val": val_unknown, "test": test_unknown},
        "crop_policy": info["crop_policy"],
        "identity": info["identity"],
        "dedup": {
            "exact_duplicates_removed": exact_duplicates,
            "near_duplicate_rule": f"horizontal and vertical dHash (64-bit each), "
            f"Hamming <= {near_dup_distance} on both",
            "largest_near_duplicate_group": max(Counter(roots).values()) if roots else 0,
            "near_duplicate_groups_with_multiple_images": sum(
                1 for size in Counter(roots).values() if size > 1
            ),
            "dog_groups_spanning_splits": sum(1 for v in dog_groups.values() if len(v) > 1),
        },
        "invalid_images": [s["sample_id"] for s in invalid],
        "counts": dict(sorted(counts.items())),
        "pseudo_target_version": spec.RGB_DESCRIPTOR_VERSION,
        "manual_labels": "none",
    }
    (base / "dataset.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("counts", "manifest_sha256", "taxonomy")}, indent=1))
    return summary


def load_manifest(root: Path, dataset: str) -> tuple[list[dict], dict, Path]:
    base = root / "datasets" / dataset
    summary = json.loads((base / "dataset.json").read_text())
    manifest_path = base / summary["manifest"]
    if sha256_file(manifest_path) != summary["manifest_sha256"]:
        raise SystemExit("Manifest checksum mismatch; rerun prepare-data")
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    return rows, summary, base / "extracted"
