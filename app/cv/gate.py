"""Minimal "is there exactly one dog in the user crop" gate (SSDLite320 MobileNetV3, COCO).

The gate never changes the crop: detector boxes are used only to count dogs.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import detection
from torchvision.transforms.functional import pil_to_tensor

from app.cv import spec
from app.cv.weights import verify

COCO_WEIGHTS = detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
# Two detections of the same dog are merged when one box mostly lies inside the other.
CONTAINMENT_MERGE = 0.7


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    reason: str | None
    dogs: int
    max_score: float  # internal diagnostics only, never shown to users


def _containment(a: torch.Tensor, b: torch.Tensor) -> float:
    left, top = torch.maximum(a[0], b[0]), torch.maximum(a[1], b[1])
    right, bottom = torch.minimum(a[2], b[2]), torch.minimum(a[3], b[3])
    inter = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    smaller = min(float((a[2] - a[0]) * (a[3] - a[1])), float((b[2] - b[0]) * (b[3] - b[1])))
    return float(inter) / smaller if smaller > 0 else 0.0


class DogGate:
    def __init__(self, weights_dir: Path | None, score_threshold: float):
        if not 0.0 < score_threshold < 1.0:
            raise ValueError("dog score threshold must be in (0, 1)")
        categories = COCO_WEIGHTS.meta["categories"]
        self.dog_label = categories.index("dog")
        self.threshold = score_threshold
        # weights_backbone=None prevents any implicit download of ImageNet backbone weights.
        model = detection.ssdlite320_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=len(categories)
        )
        if weights_dir is not None:
            state = torch.load(
                verify("ssdlite", weights_dir), map_location="cpu", weights_only=True
            )
            model.load_state_dict(state)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model.eval()

    @torch.inference_mode()
    def check(self, crop: Image.Image) -> GateResult:
        tensor = pil_to_tensor(crop).float().div(255.0)
        output = self.model([tensor])[0]
        keep = (output["labels"] == self.dog_label) & (output["scores"] >= self.threshold)
        boxes, scores = output["boxes"][keep], output["scores"][keep]
        max_score = (
            float(output["scores"][output["labels"] == self.dog_label].max().item())
            if (output["labels"] == self.dog_label).any()
            else 0.0
        )
        distinct: list[torch.Tensor] = []
        for box in boxes[torch.argsort(scores, descending=True)]:
            if all(_containment(box, other) < CONTAINMENT_MERGE for other in distinct):
                distinct.append(box)
        if not distinct:
            return GateResult(False, spec.NO_DOG, 0, max_score)
        if len(distinct) > 1:
            return GateResult(False, spec.MULTIPLE_DOGS, len(distinct), max_score)
        return GateResult(True, None, 1, max_score)
