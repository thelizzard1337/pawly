"""Model bundle export/load. A bundle holds trained heads plus everything needed to verify that
the runtime uses the same backbone, weights, preprocessing, descriptor and taxonomy."""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
import torchvision

from app.cv import spec
from app.cv.heads import HeadConfig, TwoHeads
from app.cv.pipeline import BUNDLE_MANIFEST as MANIFEST
from app.cv.pipeline import BUNDLE_SCHEMA as SCHEMA
from app.cv.pipeline import PipelineConfigError, read_bundle_manifest
from app.cv.weights import REGISTRY

HEADS_FILE = "heads.pt"


class BundleError(RuntimeError):
    """Missing, corrupted or incompatible bundle. Never falls back to another mode."""


def taxonomy_hash(classes: list[str]) -> str:
    return hashlib.sha256(spec.canonical_json(classes).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Bundle:
    path: Path
    manifest: dict
    heads: TwoHeads

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]

    @property
    def classes(self) -> list[str]:
        return self.manifest["taxonomy"]["classes"]

    @property
    def unknown_threshold(self) -> float:
        return float(self.manifest["unknown_threshold"])


def export_bundle(
    out_dir: Path,
    heads: TwoHeads,
    backbone_name: str,
    classes: list[str],
    taxonomy_source: str,
    unknown_threshold: float,
    training: dict,
    metrics: dict,
) -> Path:
    backbone = spec.BACKBONES[backbone_name]
    if heads.cfg.in_dim != backbone.dimension or heads.cfg.num_breeds != len(classes):
        raise BundleError("heads do not match backbone dimension or taxonomy size")
    if out_dir.exists():
        raise BundleError(f"bundle directory already exists: {out_dir}")
    if not 1 <= len(out_dir.name) <= 60:
        raise BundleError("bundle name must be 1..60 characters")
    temporary = out_dir.with_name(out_dir.name + ".partial")
    temporary.mkdir(parents=True, exist_ok=False)
    torch.save(heads.state_dict(), temporary / HEADS_FILE)
    digest = _sha256(temporary / HEADS_FILE)
    manifest = {
        "schema": SCHEMA,
        # Content-addressed: different head weights can never share a pipeline_id.
        "bundle_id": f"{out_dir.name}+{digest[:12]}",
        "model_stage": "trained_heads",
        "created_at": datetime.now(UTC).isoformat(),
        "backbone": backbone.name,
        "weights_id": backbone.weights_id,
        "weights_sha256": REGISTRY[backbone.name].sha256,
        "embedding_space_id": backbone.embedding_space_id,
        "embedding_dim": backbone.dimension,
        "preprocessing": backbone.preprocessing,
        "preprocessing_id": backbone.preprocessing_id,
        "descriptor_version": spec.RGB_DESCRIPTOR_VERSION,
        "crop_rule": spec.CROP_RULE,
        "taxonomy": {
            "source": taxonomy_source,
            "classes": classes,
            "sha256": taxonomy_hash(classes),
        },
        "heads": {"config": heads.cfg.as_dict(), "file": HEADS_FILE, "sha256": digest},
        "unknown_threshold": unknown_threshold,
        "training": training,
        "metrics": metrics,
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
    }
    (temporary / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    os.replace(temporary, out_dir)
    return out_dir


def read_manifest(path: Path) -> dict:
    try:
        return read_bundle_manifest(path)
    except PipelineConfigError as exc:
        raise BundleError(str(exc)) from exc


def validate_manifest(manifest: dict, backbone_name: str) -> None:
    backbone = spec.BACKBONES[backbone_name]
    expected = {
        "backbone": backbone.name,
        "weights_id": backbone.weights_id,
        "weights_sha256": REGISTRY[backbone.name].sha256,
        "embedding_space_id": backbone.embedding_space_id,
        "embedding_dim": backbone.dimension,
        "preprocessing_id": backbone.preprocessing_id,
        "descriptor_version": spec.RGB_DESCRIPTOR_VERSION,
        "crop_rule": spec.CROP_RULE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise BundleError(f"bundle_incompatible:{key}")
    if manifest.get("preprocessing") != backbone.preprocessing:
        raise BundleError("bundle_incompatible:preprocessing")
    bundle_id = manifest.get("bundle_id")
    heads_sha = (manifest.get("heads") or {}).get("sha256", "")
    if not isinstance(bundle_id, str) or not bundle_id.endswith("+" + heads_sha[:12]):
        raise BundleError("bundle_incompatible:bundle_id")
    taxonomy = manifest.get("taxonomy") or {}
    classes = taxonomy.get("classes")
    if not isinstance(classes, list) or not classes or len(set(classes)) != len(classes):
        raise BundleError("bundle_incompatible:taxonomy")
    if taxonomy.get("sha256") != taxonomy_hash(classes):
        raise BundleError("bundle_incompatible:taxonomy_hash")
    threshold = manifest.get("unknown_threshold")
    if not isinstance(threshold, int | float) or not 0.0 <= threshold <= 1.0:
        raise BundleError("bundle_incompatible:unknown_threshold")


def load_bundle(path: Path, backbone_name: str) -> Bundle:
    manifest = read_manifest(path)
    validate_manifest(manifest, backbone_name)
    head_info = manifest["heads"]
    heads_path = path / head_info["file"]
    if heads_path.name != HEADS_FILE or not heads_path.is_file():
        raise BundleError("bundle_heads_missing")
    if _sha256(heads_path) != head_info["sha256"]:
        raise BundleError("bundle_heads_checksum_mismatch")
    try:
        cfg = HeadConfig(**head_info["config"])
    except TypeError as exc:
        raise BundleError("bundle_incompatible:head_config") from exc
    if cfg.in_dim != manifest["embedding_dim"] or cfg.num_breeds != len(
        manifest["taxonomy"]["classes"]
    ):
        raise BundleError("bundle_incompatible:head_shape")
    if cfg.color_bins != spec.RGB_HIST_DIM:
        raise BundleError("bundle_incompatible:color_bins")
    heads = TwoHeads(cfg)
    try:
        state = torch.load(heads_path, map_location="cpu", weights_only=True)
        heads.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise BundleError("bundle_incompatible:state_dict") from exc
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    heads.eval()
    return Bundle(path, manifest, heads)
