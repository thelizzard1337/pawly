"""Frozen-backbone feature cache.

Uses exactly the runtime path: normalize_image -> load_canonical -> apply_crop ->
rgb_descriptor (crop pixels) and make_preprocess -> FrozenEncoder (eval, no grad). Because the
backbone is frozen, in eval mode and the transforms are deterministic, cached features equal what
the frozen backbone would produce inside every training step.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.cv import image_ops, spec
from app.cv.backbones import build_encoder, make_preprocess
from app.modules.media.images import normalize_image

MAX_PIXELS = 20_000_000


def load_sample(row: dict, extracted: Path, preprocess):
    """Shared by the cache and by the train/inference parity test."""
    raw = (extracted / row["rel_path"]).read_bytes()
    normalized = normalize_image(raw, MAX_PIXELS)
    if normalized.sha256 != row["canonical_sha256"]:
        raise RuntimeError(f"canonical hash changed for {row['sample_id']}")
    image = image_ops.load_canonical(normalized.content)
    crop, _ = image_ops.apply_crop(image, spec.bbox_or_none(row.get("bbox")))
    histogram = image_ops.rgb_descriptor(crop).histogram
    return preprocess(crop), np.asarray(histogram, dtype=np.float32)


class ManifestImages(Dataset):
    def __init__(self, rows, extracted: Path, backbone: spec.BackboneSpec):
        self.rows, self.extracted = rows, extracted
        self.preprocess = make_preprocess(backbone)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        tensor, histogram = load_sample(self.rows[index], self.extracted, self.preprocess)
        return tensor, torch.from_numpy(histogram), index


def cache_key(summary: dict, backbone: spec.BackboneSpec, splits: tuple[str, ...]) -> str:
    payload = {
        "manifest_sha256": summary["manifest_sha256"],
        "embedding_space_id": backbone.embedding_space_id,
        "descriptor": spec.RGB_DESCRIPTOR_VERSION,
        "crop_rule": spec.CROP_RULE,
        "splits": sorted(splits),
    }
    return hashlib.sha256(spec.canonical_json(payload).encode()).hexdigest()[:16]


def load_or_extract(
    root: Path,
    dataset: str,
    rows,
    summary,
    extracted: Path,
    backbone_name: str,
    weights_dir: Path,
    splits: tuple[str, ...],
    workers: int,
    batch_size: int,
    device: str,
) -> dict:
    backbone = spec.BACKBONES[backbone_name]
    selected = [r for r in rows if r["split"] in splits]
    key = cache_key(summary, backbone, splits)
    directory = root / "features" / dataset
    path = directory / f"{backbone_name}-{key}.npz"
    if path.is_file():
        data = np.load(path, allow_pickle=False)
        meta = json.loads(path.with_suffix(".json").read_text())
        return {
            "ids": list(data["ids"]),
            "features": data["features"],
            "rgb": data["rgb"],
            "meta": meta,
            "path": str(path),
        }
    encoder = build_encoder(backbone_name, weights_dir).to(device)
    loader = DataLoader(
        ManifestImages(selected, extracted, backbone),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
    )
    features = np.zeros((len(selected), backbone.dimension), dtype=np.float32)
    rgb = np.zeros((len(selected), spec.RGB_HIST_DIM), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch, histograms, indices in loader:
            out = encoder(batch.to(device)).cpu().numpy()
            features[indices.numpy()] = out
            rgb[indices.numpy()] = histograms.numpy()
    elapsed = time.perf_counter() - started
    if not np.isfinite(features).all():
        raise RuntimeError("non-finite features")
    directory.mkdir(parents=True, exist_ok=True)
    ids = np.array([r["sample_id"] for r in selected])
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary, ids=ids, features=features, rgb=rgb)
    meta = {
        "dataset": dataset,
        "backbone": backbone_name,
        "weights_id": backbone.weights_id,
        "embedding_space_id": backbone.embedding_space_id,
        "preprocessing": backbone.preprocessing,
        "descriptor": spec.RGB_DESCRIPTOR_VERSION,
        "crop_rule": spec.CROP_RULE,
        "manifest_sha256": summary["manifest_sha256"],
        "splits": sorted(splits),
        "samples": len(selected),
        "extraction_seconds": round(elapsed, 2),
        "device": device,
        "torch_threads": torch.get_num_threads(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    temporary.replace(path)
    return {"ids": list(ids), "features": features, "rgb": rgb, "meta": meta, "path": str(path)}
