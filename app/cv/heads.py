"""Two trainable heads on top of a frozen shared backbone: breed classes and RGB histogram.

Breed head: K logits, softmax over the known taxonomy (single-label CrossEntropy).
Color head: 64 logits, softmax = predicted RGB histogram (a distribution that sums to 1),
trained with KL(target || predicted) against the algorithmic RGB histogram of the same crop.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from app.cv import spec

UNKNOWN_LABEL = -1  # samples without a known-taxonomy label (unknown breed, mixed breed)


@dataclass(frozen=True)
class HeadConfig:
    in_dim: int
    num_breeds: int
    color_bins: int = spec.RGB_HIST_DIM
    hidden: int = 0  # 0 -> Linear heads; >0 -> Linear-ReLU-Dropout-Linear
    dropout: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _head(cfg: HeadConfig, out_dim: int) -> nn.Module:
    if cfg.hidden <= 0:
        return nn.Linear(cfg.in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(cfg.in_dim, cfg.hidden),
        nn.ReLU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.hidden, out_dim),
    )


class TwoHeads(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        self.breed = _head(cfg, cfg.num_breeds)
        self.color = _head(cfg, cfg.color_bins)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.breed(features), self.color(features)


def breed_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """CrossEntropy over samples with a known label only; unknown/mixed are masked out."""
    known = labels != UNKNOWN_LABEL
    if not bool(known.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[known], labels[known])


def color_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of KL(target || softmax(logits)), with 0 * log 0 = 0.

    Equals soft-target cross-entropy minus the (constant) target entropy, so both have the same
    gradient with respect to the logits.
    """
    log_pred = F.log_softmax(logits, dim=1)
    return F.kl_div(log_pred, target, reduction="batchmean")


def total_loss(breed_logits, color_logits, labels, targets, breed_weight=1.0, color_weight=1.0):
    b = breed_loss(breed_logits, labels)
    c = color_loss(color_logits, targets)
    return breed_weight * b + color_weight * c, b.detach(), c.detach()


class TwoHeadModel(nn.Module):
    """Frozen encoder + trainable heads. ``train()`` never switches the encoder to train mode."""

    def __init__(self, encoder: nn.Module, heads: TwoHeads):
        super().__init__()
        self.encoder, self.heads = encoder, heads
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # BatchNorm statistics, dropout and stochastic depth of the backbone stay in eval mode.
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor):
        with torch.no_grad():
            features = self.encoder(images)
        return self.heads(features)


def make_optimizer(heads: TwoHeads, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in heads.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )


def train_step(model, optimizer, inputs, labels, targets, breed_weight=1.0, color_weight=1.0):
    """One optimisation step. ``model`` is a TwoHeadModel (images) or TwoHeads (cached features)."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    breed_logits, color_logits = model(inputs)
    loss, b, c = total_loss(breed_logits, color_logits, labels, targets, breed_weight, color_weight)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), float(b), float(c)


def breed_output(logits: torch.Tensor, classes: list[str], threshold: float, top_k: int = 3):
    """Soft breed signal: top-k known classes and an "unknown" flag from max-softmax threshold."""
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=min(top_k, probs.numel()))
    return {
        "top": [
            {"class": classes[int(i)], "p": round(float(v), 6)}
            for v, i in zip(values, indices, strict=True)
        ],
        "unknown": bool(float(values[0]) < threshold),
        "threshold": threshold,
        "distribution": [round(float(v), 6) for v in probs],
    }
