"""Torch-free description of the active CV pipeline (used by worker, CLI and status)."""

import json
from dataclasses import dataclass
from pathlib import Path

from app.cv import spec

PROVIDER = "pawly-cv"
MOCK_PROVIDER = "mock-v1"
BUNDLE_SCHEMA = "pawly-cv-bundle/v1"
BUNDLE_MANIFEST = "manifest.json"

# Ranking versions. embedding-v1 is the baseline; the others are experimental rerankers whose
# weights are configuration, not tuned or proven values.
FUSIONS = ("embedding-v1", "embedding-rgb-v1", "embedding-rgb-heads-v1")


class PipelineConfigError(RuntimeError):
    """The configured CV mode cannot run (e.g. trained_heads without a valid bundle)."""


def read_bundle_manifest(path: Path) -> dict:
    try:
        manifest = json.loads((path / BUNDLE_MANIFEST).read_text())
    except FileNotFoundError as exc:
        raise PipelineConfigError(f"bundle_missing:{path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineConfigError("bundle_manifest_unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != BUNDLE_SCHEMA:
        raise PipelineConfigError("bundle_schema_unsupported")
    if not isinstance(manifest.get("bundle_id"), str) or not manifest["bundle_id"]:
        raise PipelineConfigError("bundle_id_missing")
    return manifest


@dataclass(frozen=True)
class PipelineSpec:
    mode: str
    backbone: spec.BackboneSpec
    gate_threshold: float
    bundle_id: str | None
    fusion: str

    @property
    def embedding_space_id(self) -> str:
        return self.backbone.embedding_space_id

    @property
    def gate_id(self) -> str:
        return spec.gate_id(self.gate_threshold)

    @property
    def pipeline_id(self) -> str:
        heads = f"heads:{self.bundle_id}" if self.bundle_id else "heads:none"
        return "|".join(
            (
                self.embedding_space_id,
                spec.RGB_DESCRIPTOR_VERSION,
                self.gate_id,
                spec.CROP_RULE,
                heads,
            )
        )

    @property
    def provider(self) -> str:
        return PROVIDER

    @property
    def model_version(self) -> str:
        return self.bundle_id or f"{self.backbone.name}.{self.backbone.weights_id}"

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "provider": self.provider,
            "pipeline_id": self.pipeline_id,
            "embedding_space_id": self.embedding_space_id,
            "backbone": self.backbone.name,
            "weights_id": self.backbone.weights_id,
            "embedding_dim": self.backbone.dimension,
            "preprocessing_id": self.backbone.preprocessing_id,
            "descriptor_version": spec.RGB_DESCRIPTOR_VERSION,
            "gate_id": self.gate_id,
            "crop_rule": spec.CROP_RULE,
            "bundle_id": self.bundle_id,
            "fusion": self.fusion,
        }


def resolve(settings) -> PipelineSpec:
    if settings.cv_mode == "mock":
        raise PipelineConfigError("mock mode has no CV pipeline")
    backbone = spec.BACKBONES[settings.cv_backbone]
    bundle_id = None
    if settings.cv_mode == "trained_heads":
        if settings.cv_bundle_dir is None:
            raise PipelineConfigError("bundle_not_configured")
        manifest = read_bundle_manifest(settings.cv_bundle_dir)
        if manifest.get("backbone") != backbone.name:
            raise PipelineConfigError("bundle_incompatible:backbone")
        bundle_id = manifest["bundle_id"]
    pipeline = PipelineSpec(
        settings.cv_mode, backbone, settings.cv_dog_score_threshold, bundle_id, settings.cv_fusion
    )
    if len(pipeline.pipeline_id) > 200 or (bundle_id and len(bundle_id) > 120):
        raise PipelineConfigError("pipeline_id_too_long")
    return pipeline
