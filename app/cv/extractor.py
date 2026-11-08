"""Attribute extraction shared by listing indexing and search queries.

canonical image -> user crop -> dog gate -> RGB descriptor (on the crop) and backbone
preprocessing -> L2-normalized pooled embedding -> optional trained heads.
"""

import time
from dataclasses import dataclass, field

import torch

from app.cv import image_ops, spec
from app.cv.backbones import build_encoder, l2_normalize, make_preprocess
from app.cv.bundle import Bundle, load_bundle
from app.cv.gate import DogGate
from app.cv.heads import breed_output
from app.cv.pipeline import PipelineSpec, resolve


@dataclass
class Extraction:
    crop: spec.CropSpec | None
    rejected: str | None = None
    embedding: list[float] | None = None
    rgb_histogram: list[float] | None = None
    dominant_colors: list[dict] | None = None
    breed_prediction: dict | None = None
    color_prediction: list[float] | None = None
    timings_ms: dict = field(default_factory=dict)


class FeatureExtractor:
    """Owns the loaded models. Not thread-safe: call from one dedicated thread."""

    def __init__(
        self,
        pipeline: PipelineSpec,
        encoder,
        gate: DogGate,
        bundle: Bundle | None,
    ):
        if (pipeline.mode == "trained_heads") != (bundle is not None):
            raise ValueError("trained_heads mode requires exactly one loaded bundle")
        self.pipeline, self.encoder, self.gate, self.bundle = pipeline, encoder, gate, bundle
        self.preprocess = make_preprocess(pipeline.backbone)

    @classmethod
    def from_settings(cls, settings) -> "FeatureExtractor":
        torch.set_num_threads(settings.cv_torch_threads)
        pipeline = resolve(settings)
        encoder = build_encoder(pipeline.backbone.name, settings.cv_weights_dir)
        gate = DogGate(settings.cv_weights_dir, settings.cv_dog_score_threshold)
        bundle = None
        if pipeline.mode == "trained_heads":
            bundle = load_bundle(settings.cv_bundle_dir, pipeline.backbone.name)
            if bundle.bundle_id != pipeline.bundle_id:
                raise ValueError("bundle changed while loading")
        return cls(pipeline, encoder, gate, bundle)

    @torch.inference_mode()
    def extract(self, raw: bytes, bbox: spec.BBox | None) -> Extraction:
        started = time.perf_counter()
        timings = {}
        try:
            image = image_ops.load_canonical(raw)
            crop_image, crop = image_ops.apply_crop(image, bbox)
        except image_ops.ImageRejected as exc:
            return Extraction(None, rejected=exc.reason)
        gate = self.gate.check(crop_image)
        timings["gate"] = round((time.perf_counter() - started) * 1000, 2)
        if not gate.accepted:
            return Extraction(crop, rejected=gate.reason, timings_ms=timings)
        descriptor = image_ops.rgb_descriptor(crop_image)
        tensor = self.preprocess(crop_image).unsqueeze(0)
        features = self.encoder(tensor)
        if features.shape != (1, self.pipeline.backbone.dimension):
            raise RuntimeError(f"unexpected embedding shape {tuple(features.shape)}")
        embedding = l2_normalize(features)[0]
        result = Extraction(
            crop,
            embedding=[float(v) for v in embedding],
            rgb_histogram=descriptor.histogram,
            dominant_colors=descriptor.dominant,
        )
        if self.bundle is not None:
            breed_logits, color_logits = self.bundle.heads(features)
            result.breed_prediction = breed_output(
                breed_logits[0], self.bundle.classes, self.bundle.unknown_threshold
            )
            result.color_prediction = [
                round(float(v), 8) for v in torch.softmax(color_logits[0], dim=-1)
            ]
        timings["total"] = round((time.perf_counter() - started) * 1000, 2)
        result.timings_ms = timings
        return result
