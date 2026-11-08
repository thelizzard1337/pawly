"""python -m app.ml {prepare-data,train,evaluate,export,check-bundle}"""

import argparse
import os
from pathlib import Path


def main(argv=None):
    root_default = Path(os.environ.get("ML_ROOT", "/data/ml"))
    weights_default = Path(os.environ.get("CV_WEIGHTS_DIR", "/opt/pawly/weights"))
    parser = argparse.ArgumentParser(prog="python -m app.ml", description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--weights-dir", type=Path, default=weights_default)
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare-data", help="download/verify/extract + manifest")
    prep.add_argument("dataset", choices=["oxford-pets", "stanford-dogs"])
    prep.add_argument("--archives-dir", type=Path, help="pre-downloaded archives to copy")
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--holdout-breeds", type=int, default=4)
    prep.add_argument("--val-fraction", type=float, default=0.15)
    prep.add_argument("--near-dup-distance", type=int, default=5)
    prep.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    prep.add_argument("--accept-terms", help="required for stanford-dogs")

    def feature_args(p):
        p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--device", default="cpu", help="cpu (default), cuda or auto")

    tr = commands.add_parser("train", help="head-only training on frozen features")
    tr.add_argument("--dataset", default="oxford-pets")
    tr.add_argument(
        "--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"]
    )
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--train-batch", type=int, default=128)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--weight-decay", type=float, default=1e-4)
    tr.add_argument("--hidden", type=int, default=0)
    tr.add_argument("--dropout", type=float, default=0.0)
    tr.add_argument("--breed-weight", type=float, default=1.0)
    tr.add_argument("--color-weight", type=float, default=1.0)
    tr.add_argument("--known-tpr", type=float, default=0.95)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--run-name")
    feature_args(tr)

    ev = commands.add_parser("evaluate", help="test-split metrics for a run")
    ev.add_argument("--run", type=Path, required=True)
    ev.add_argument("--ks", default="1,5,10,20,50")
    ev.add_argument("--fusion-rgb-weight", type=float, default=0.2)
    ev.add_argument("--fusion-heads-weight", type=float, default=0.1)
    ev.add_argument("--gate-samples", type=int, default=200)
    ev.add_argument("--dog-threshold", type=float, default=0.5)
    feature_args(ev)

    ex = commands.add_parser("export", help="export a run as a verified model bundle")
    ex.add_argument("--run", type=Path, required=True)
    ex.add_argument("--name")

    ck = commands.add_parser("check-bundle", help="validate a bundle against the runtime")
    ck.add_argument("path", type=Path)
    ck.add_argument("--backbone", default="efficientnet_b0")

    args = parser.parse_args(argv)
    if args.command == "prepare-data":
        from app.ml.datasets import prepare

        prepare(
            args.dataset,
            args.root,
            args.archives_dir,
            args.seed,
            args.holdout_breeds,
            args.val_fraction,
            args.near_dup_distance,
            args.workers,
            args.accept_terms,
        )
    elif args.command == "train":
        from app.ml.experiment import train

        train(args)
    elif args.command == "evaluate":
        from app.ml.experiment import evaluate

        if not args.run.is_absolute():
            args.run = args.root / "runs" / args.run
        evaluate(args)
    elif args.command == "export":
        from app.ml.experiment import export

        if not args.run.is_absolute():
            args.run = args.root / "runs" / args.run
        export(args)
    else:
        from app.cv.bundle import load_bundle

        bundle = load_bundle(args.path, args.backbone)
        print(
            f"valid bundle {bundle.bundle_id}: {len(bundle.classes)} classes, "
            f"threshold {bundle.unknown_threshold}"
        )


if __name__ == "__main__":
    main()
