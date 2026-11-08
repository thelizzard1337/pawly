"""Head-only training, evaluation and bundle export on cached frozen features."""

import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from app.cv import image_ops, spec
from app.cv.bundle import export_bundle, load_bundle
from app.cv.heads import UNKNOWN_LABEL, HeadConfig, TwoHeads, make_optimizer, train_step
from app.ml import metrics
from app.ml.datasets import load_manifest
from app.ml.features import load_or_extract

SPLITS = ("train", "val", "test")


def _device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def _arrays(cache, rows_by_id, classes):
    index = {c: i for i, c in enumerate(classes)}
    labels, splits, statuses, breeds, groups = [], [], [], [], []
    for sample_id in cache["ids"]:
        row = rows_by_id[sample_id]
        labels.append(index.get(row["class_name"], UNKNOWN_LABEL))
        splits.append(row["split"])
        statuses.append(row["label_status"])
        breeds.append(row["breed"])
        groups.append(row["group_id"])
    return {
        "labels": np.array(labels, dtype=np.int64),
        "splits": np.array(splits),
        "status": np.array(statuses),
        "breeds": np.array(breeds),
        "groups": np.array(groups),
    }


def _load(args):
    rows, summary, extracted = load_manifest(args.root, args.dataset)
    cache = load_or_extract(
        args.root,
        args.dataset,
        rows,
        summary,
        extracted,
        args.backbone,
        args.weights_dir,
        SPLITS,
        args.workers,
        args.batch_size,
        _device(args.device),
    )
    classes = summary["taxonomy"]
    arrays = _arrays(cache, {r["sample_id"]: r for r in rows}, classes)
    return rows, summary, cache, classes, arrays


@torch.no_grad()
def _predict(heads, features, device):
    heads.eval()
    tensor = torch.from_numpy(features).to(device)
    breed, color = heads(tensor)
    return torch.softmax(breed, 1).cpu().numpy(), torch.softmax(color, 1).cpu().numpy()


def train(args) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _device(args.device)
    rows, summary, cache, classes, arrays = _load(args)
    features, rgb = cache["features"], cache["rgb"]
    train_mask = arrays["splits"] == "train"
    # Unknown/mixed samples never get a fake class: only known labels reach the breed loss.
    if (arrays["labels"][train_mask] == UNKNOWN_LABEL).any():
        raise RuntimeError("held-out (unknown) breeds leaked into the training split")
    val_known = (arrays["splits"] == "val") & (arrays["labels"] != UNKNOWN_LABEL)
    cfg = HeadConfig(
        spec.BACKBONES[args.backbone].dimension,
        len(classes),
        hidden=args.hidden,
        dropout=args.dropout,
    )
    heads = TwoHeads(cfg).to(device)
    optimizer = make_optimizer(heads, args.lr, args.weight_decay)
    x_train = torch.from_numpy(features[train_mask]).to(device)
    y_train = torch.from_numpy(arrays["labels"][train_mask]).to(device)
    t_train = torch.from_numpy(rgb[train_mask].astype(np.float32)).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    history, best = [], None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(x_train), generator=generator)
        losses = []
        for start in range(0, len(order), args.train_batch):
            idx = order[start : start + args.train_batch].to(device)
            losses.append(
                train_step(
                    heads,
                    optimizer,
                    x_train[idx],
                    y_train[idx],
                    t_train[idx],
                    args.breed_weight,
                    args.color_weight,
                )
            )
        breed_p, color_p = _predict(heads, features[val_known], device)
        val_cls = metrics.classification_metrics(
            arrays["labels"][val_known], breed_p.argmax(1), classes
        )
        val_color = metrics.histogram_metrics(rgb[val_known].astype(np.float64), color_p)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean([x[0] for x in losses])),
            "train_breed_loss": float(np.mean([x[1] for x in losses])),
            "train_color_kl": float(np.mean([x[2] for x in losses])),
            "val_macro_recall": val_cls["macro_recall"],
            "val_accuracy": val_cls["accuracy"],
            "val_color_kl": val_color["kl_mean"],
        }
        history.append(record)
        print(json.dumps(record))
        key = (record["val_macro_recall"], -record["val_color_kl"])
        if best is None or key > best[0]:
            best = (
                key,
                epoch,
                {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()},
            )
    train_seconds = time.perf_counter() - started
    heads.load_state_dict(best[2])
    # Unknown threshold on VALIDATION only: keep known_tpr of known samples (recall first).
    val_mask = arrays["splits"] == "val"
    breed_p, _ = _predict(heads, features[val_mask], device)
    max_p = breed_p.max(1)
    known = arrays["labels"][val_mask] != UNKNOWN_LABEL
    threshold = float(np.quantile(max_p[known], 1.0 - args.known_tpr))
    unknown_rejected = float((max_p[~known] < threshold).mean()) if (~known).any() else None
    run_id = args.run_name or (
        f"{args.dataset}-{args.backbone}-s{args.seed}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    run_dir = args.root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save(heads.cpu().state_dict(), run_dir / "heads.pt")
    trainable = sum(p.numel() for p in heads.parameters() if p.requires_grad)
    config = {
        "run_id": run_id,
        "dataset": args.dataset,
        "dataset_manifest_sha256": summary["manifest_sha256"],
        "feature_cache": cache["meta"],
        "backbone": args.backbone,
        "embedding_space_id": spec.BACKBONES[args.backbone].embedding_space_id,
        "head_config": cfg.as_dict(),
        "trainable_parameters": trainable,
        "backbone_trainable_parameters": 0,
        "classes": classes,
        "heldout_unknown_breeds": summary["heldout_unknown_breeds"],
        "losses": {
            "breed": "CrossEntropy over known-taxonomy samples (unknown masked)",
            "color": "KL(target RGB histogram || softmax(color logits)), batchmean",
            "total": f"{args.breed_weight}*breed + {args.color_weight}*color",
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "train_batch": args.train_batch,
            "seed": args.seed,
            "optimizer": "AdamW",
        },
        "selection": "best epoch by validation macro recall (tie: lower validation color KL)",
        "best_epoch": best[1],
        "unknown_threshold": threshold,
        "unknown_threshold_rule": f"max-softmax quantile keeping {args.known_tpr:.0%} of "
        "validation known samples",
        "validation_unknown_rejection_at_threshold": unknown_rejected,
        "split_sizes": {s: int((arrays["splits"] == s).sum()) for s in SPLITS},
        "hardware": {
            "device": device,
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "train_seconds": round(train_seconds, 2),
        "versions": {"torch": torch.__version__},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Run saved: {run_dir}")
    return run_dir


def _load_run(run_dir: Path):
    config = json.loads((run_dir / "config.json").read_text())
    heads = TwoHeads(HeadConfig(**config["head_config"]))
    heads.load_state_dict(torch.load(run_dir / "heads.pt", map_location="cpu", weights_only=True))
    return config, heads.eval()


def _gate_eval(rows, extracted, weights_dir, threshold, samples, seed):
    from app.cv.gate import DogGate
    from app.ml.features import MAX_PIXELS
    from app.modules.media.images import normalize_image

    gate = DogGate(weights_dir, threshold)
    rng = np.random.default_rng(seed)
    result = {}
    for name, pool in (
        ("dogs_test", [r for r in rows if r["split"] == "test"]),
        ("cats_non_dog", [r for r in rows if r["split"] == "gate_negative"]),
    ):
        chosen = [
            pool[i] for i in sorted(rng.choice(len(pool), min(samples, len(pool)), replace=False))
        ]
        reasons, timings = {}, []
        for row in chosen:
            raw = normalize_image((extracted / row["rel_path"]).read_bytes(), MAX_PIXELS)
            image = image_ops.load_canonical(raw.content)
            started = time.perf_counter()
            outcome = gate.check(image)
            timings.append((time.perf_counter() - started) * 1000)
            key = "accepted" if outcome.accepted else outcome.reason
            reasons[key] = reasons.get(key, 0) + 1
        result[name] = {
            "samples": len(chosen),
            "accepted_rate": reasons.get("accepted", 0) / max(len(chosen), 1),
            "outcomes": reasons,
            "gate_ms_median": float(np.median(timings)) if timings else None,
        }
    result["dog_recall"] = result["dogs_test"]["accepted_rate"]
    result["non_dog_false_acceptance"] = result["cats_non_dog"]["accepted_rate"]
    result["note"] = "Cats from Oxford-IIIT Pet are the only labelled non-dog images used."
    return result


def evaluate(args) -> dict:
    run_dir = args.run
    config, heads = _load_run(run_dir)
    args.dataset, args.backbone = config["dataset"], config["backbone"]
    rows, summary, cache, classes, arrays = _load(args)
    if classes != config["classes"]:
        raise SystemExit("taxonomy of the run does not match the dataset manifest")
    features, rgb = cache["features"], cache["rgb"].astype(np.float64)
    test = arrays["splits"] == "test"
    known = arrays["labels"] != UNKNOWN_LABEL
    breed_p, color_p = _predict(heads, features, "cpu")
    report = {"run_id": config["run_id"], "split": "test", "parameters_fixed_before_test": True}
    tk = test & known
    report["breed_head"] = metrics.classification_metrics(
        arrays["labels"][tk], breed_p[tk].argmax(1), classes
    )
    t = config["unknown_threshold"]
    max_p = breed_p.max(1)
    report["unknown_threshold"] = {
        "threshold": t,
        "chosen_on": "validation",
        "test_known_accepted": float((max_p[tk] >= t).mean()),
        "test_heldout_rejected": float((max_p[test & ~known] < t).mean())
        if (test & ~known).any()
        else None,
        "test_heldout_breeds": summary["heldout_unknown_breeds"]["test"],
    }
    train_mean = rgb[arrays["splits"] == "train"].mean(0, keepdims=True)
    report["color_head_pseudo_targets"] = {
        "target": f"{spec.RGB_DESCRIPTOR_VERSION} of the same crop (algorithmic pseudo-target)",
        "head": metrics.histogram_metrics(rgb[test], color_p[test]),
        "baseline_train_mean_histogram": metrics.histogram_metrics(
            rgb[test], np.repeat(train_mean, test.sum(), axis=0)
        ),
        "note": "Reproduction of a crop RGB histogram, not coat-colour recognition accuracy.",
    }
    # Breed-level retrieval on the test split (all test dogs, held-out breeds included).
    emb = features[test] / np.linalg.norm(features[test], axis=1, keepdims=True)
    ids = [cache["ids"][i] for i in np.nonzero(test)[0]]
    breeds, groups = arrays["breeds"][test], arrays["groups"][test]
    relevance = breeds[:, None] == breeds[None, :]
    excluded = groups[:, None] == groups[None, :]  # self and near-duplicates never ranked
    task = metrics.RetrievalTask(
        ids,
        ids,
        relevance,
        excluded,
        {
            "dataset": config["dataset"],
            "split": "test",
            "relevance": "same dataset breed label (attribute/breed retrieval, NOT same dog)",
            "embedding_space_id": config["embedding_space_id"],
        },
    )
    cosine = emb @ emb.T
    w_rgb, w_heads = args.fusion_rgb_weight, args.fusion_heads_weight
    hi = metrics.histogram_intersection_matrix(rgb[test], rgb[test])
    methods = {
        "A_embedding-v1": cosine,
        "B_embedding-rgb-v1": (1 - w_rgb) * (1 + cosine) / 2 + w_rgb * hi,
        "C_embedding-rgb-heads-v1": (1 - w_rgb - w_heads) * (1 + cosine) / 2
        + w_rgb * hi
        + w_heads
        * 0.5
        * (
            metrics.bhattacharyya_matrix(breed_p[test], breed_p[test])
            + metrics.histogram_intersection_matrix(color_p[test], color_p[test])
        ),
    }
    ks = [int(k) for k in args.ks.split(",")]
    heldout_q = ~known[test]
    report["retrieval_breed_level"] = {
        "definition": task.description,
        "fusion_weights": {"rgb": w_rgb, "heads": w_heads, "tuned": False},
        "methods": {},
    }
    for name, scores in methods.items():
        entry = {"all_queries": metrics.retrieval_metrics(task, scores, ks)}
        for subset, mask in (
            ("known_breed_queries", ~heldout_q),
            ("heldout_breed_queries", heldout_q),
        ):
            sub = metrics.RetrievalTask(
                [i for i, m in zip(ids, mask, strict=True) if m],
                ids,
                relevance[mask],
                excluded[mask],
                task.description,
            )
            entry[subset] = metrics.retrieval_metrics(sub, scores[mask], ks)
        report["retrieval_breed_level"]["methods"][name] = entry
    report["retrieval_same_dog"] = {
        "status": "unavailable",
        "reason": "Oxford-IIIT Pet has no individual-animal identity labels. Breed/attribute "
        "proximity was measured; same-dog search quality was NOT measured.",
    }
    started = time.perf_counter()
    for row in emb[: min(200, len(emb))]:
        np.argsort(-(emb @ row), kind="stable")[:50]
    report["latency"] = {
        "feature_cache_extraction_seconds": cache["meta"]["extraction_seconds"],
        "feature_cache_samples": cache["meta"]["samples"],
        "offline_exact_cosine_ms_per_query": round(
            (time.perf_counter() - started) * 1000 / max(1, min(200, len(emb))), 4
        ),
        "gallery_size": len(emb),
    }
    if args.gate_samples > 0:
        report["dog_gate"] = _gate_eval(
            rows,
            args.root / "datasets" / config["dataset"] / "extracted",
            args.weights_dir,
            args.dog_threshold,
            args.gate_samples,
            config["hyperparameters"]["seed"],
        )
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(_summary(report), indent=2))
    return report


def _summary(report: dict) -> dict:
    retrieval = {
        name: {k: v for k, v in entry["all_queries"]["at_k"].items()}
        for name, entry in report["retrieval_breed_level"]["methods"].items()
    }
    return {
        "breed_head": {k: report["breed_head"][k] for k in ("samples", "accuracy", "macro_recall")},
        "unknown_threshold": report["unknown_threshold"],
        "color_head": report["color_head_pseudo_targets"]["head"],
        "color_baseline": report["color_head_pseudo_targets"]["baseline_train_mean_histogram"],
        "retrieval_breed_level": retrieval,
        "dog_gate": {
            k: report.get("dog_gate", {}).get(k) for k in ("dog_recall", "non_dog_false_acceptance")
        },
    }


def export(args) -> Path:
    run_dir = args.run
    config, heads = _load_run(run_dir)
    metrics_path = run_dir / "metrics.json"
    summary = _summary(json.loads(metrics_path.read_text())) if metrics_path.exists() else None
    name = args.name or config["run_id"]
    out = args.root / "bundles" / name
    export_bundle(
        out,
        heads,
        config["backbone"],
        config["classes"],
        f"{config['dataset']} (dataset breed labels; held-out breeds excluded)",
        config["unknown_threshold"],
        training={
            k: config[k]
            for k in (
                "run_id",
                "dataset",
                "dataset_manifest_sha256",
                "trainable_parameters",
                "best_epoch",
                "hyperparameters",
                "losses",
                "unknown_threshold_rule",
            )
        },
        metrics=summary or {"note": "not evaluated before export"},
    )
    bundle = load_bundle(out, config["backbone"])  # roundtrip check
    print(f"Bundle exported and verified: {out} ({bundle.bundle_id})")
    print(
        f"Activate: CV_MODE=trained_heads CV_BUNDLE_DIR=/data/ml/bundles/{name} "
        f"CV_BACKBONE={config['backbone']}"
    )
    return out
