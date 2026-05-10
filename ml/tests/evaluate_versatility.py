"""Versatility evaluation for the multimodal game classifier.

Loads a trained checkpoint and evaluates it on three buckets defined in an
experiment config:

* ``trained``    — the held-out 20% test split of the zips used for training
* ``similar``    — held-out testers similar to the training distribution
* ``out_of_box`` — held-out testers unlike the training distribution

Writes a Markdown table and a CSV to ``tests/results/versatility_<name>.{md,csv}``.

Usage::

    python tests/evaluate_versatility.py experiments/games_only_v1.json \
        --model models/versatility_games_only_v1.pth
"""

import argparse
import csv
import json
import logging
import re
import sys
from glob import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multimodal_dataset import (  # noqa: E402
    _LABEL_MAP,
    MultimodalGameDataset,
    split_multimodal_dataset,
)
from src.multimodal_model import MultimodalGameClassifier  # noqa: E402

logger = logging.getLogger(__name__)

_TESTER_RE = re.compile(r"_([a-z0-9_()]+)\.zip$")
_LABEL_NAMES = sorted(_LABEL_MAP, key=_LABEL_MAP.get)  # index-aligned


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _expand(patterns: list[str]) -> list[str]:
    out: set[str] = set()
    for pat in patterns:
        matches = glob(pat)
        if not matches and Path(pat).exists():
            matches = [pat]
        if not matches:
            logger.warning("No matches for pattern: %s", pat)
        out.update(matches)
    return sorted(out)


def _tester_of(zip_path: str) -> str:
    m = _TESTER_RE.search(Path(zip_path).name)
    return m.group(1) if m else "unknown"


def _load_model(ckpt_path: Path, device: torch.device) -> MultimodalGameClassifier:
    model = MultimodalGameClassifier(num_classes=len(_LABEL_MAP))
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def _evaluate_subset(
    model: MultimodalGameClassifier,
    subset: Subset,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[list[int], list[int], list[str]]:
    """Run inference on a Subset; return (y_true, y_pred, zip_paths_per_sample)."""
    underlying: MultimodalGameDataset = subset.dataset  # type: ignore[assignment]
    zip_per_sample = [underlying._index[i][0] for i in subset.indices]

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, tabular, labels in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            outputs = model(images, tabular)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())

    return y_true, y_pred, zip_per_sample


def _rows_for_bucket(
    bucket: str,
    y_true: list[int],
    y_pred: list[int],
    zip_per_sample: list[str],
) -> list[dict]:
    """Produce per-tester rows plus one ALL row for a bucket."""
    rows: list[dict] = []

    def _metrics(yt: list[int], yp: list[int]) -> dict:
        n = len(yt)
        overall = sum(1 for a, b in zip(yt, yp) if a == b) / n if n else 0.0
        per_class: dict[str, float | str] = {}
        for idx, name in enumerate(_LABEL_NAMES):
            cls_yt = [p for t, p in zip(yt, yp) if t == idx]
            total = sum(1 for t in yt if t == idx)
            correct = sum(1 for p in cls_yt if p == idx)
            per_class[name] = (correct / total) if total > 0 else "N/A"
        return {"overall": overall, "per_class": per_class, "n": n}

    testers = sorted(set(_tester_of(z) for z in zip_per_sample))
    for tester in testers:
        mask = [z for z in zip_per_sample if _tester_of(z) == tester]
        idx_mask = [i for i, z in enumerate(zip_per_sample) if _tester_of(z) == tester]
        yt = [y_true[i] for i in idx_mask]
        yp = [y_pred[i] for i in idx_mask]
        m = _metrics(yt, yp)
        rows.append({
            "bucket": bucket,
            "tester": tester,
            "num_samples": m["n"],
            "overall_acc": m["overall"],
            **{f"{k.lower().replace(' ', '_')}_acc": v for k, v in m["per_class"].items()},
        })

    # ALL row
    m = _metrics(y_true, y_pred)
    rows.append({
        "bucket": bucket,
        "tester": "ALL",
        "num_samples": m["n"],
        "overall_acc": m["overall"],
        **{f"{k.lower().replace(' ', '_')}_acc": v for k, v in m["per_class"].items()},
    })
    return rows


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_outputs(
    out_dir: Path,
    name: str,
    rows: list[dict],
    cfg: dict,
    train_zips: list[str],
    model_path: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"versatility_{name}.md"
    csv_path = out_dir / f"versatility_{name}.csv"

    # Column order: fixed plus dynamic per-class columns derived from _LABEL_NAMES
    class_cols = [f"{n.lower().replace(' ', '_')}_acc" for n in _LABEL_NAMES]
    columns = ["bucket", "tester", "num_samples", "overall_acc", *class_cols]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: _fmt(r.get(c, "")) for c in columns})

    with md_path.open("w") as f:
        f.write(f"# Versatility evaluation — {name}\n\n")
        f.write(f"- **Model**: `{model_path}`\n")
        f.write(f"- **Seed**: {cfg['seed']} · **test_ratio**: {cfg['test_ratio']}\n")
        f.write(f"- **Trained on {len(train_zips)} zip(s)**\n")
        for bucket_key in ("similar", "out_of_box"):
            patterns = cfg.get(bucket_key) or []
            f.write(f"- **{bucket_key}**: {len(patterns)} pattern(s) — {patterns}\n")
        f.write("\n")

        buckets_seen: list[str] = []
        for r in rows:
            if r["bucket"] not in buckets_seen:
                buckets_seen.append(r["bucket"])

        for bucket in buckets_seen:
            f.write(f"## {bucket}\n\n")
            f.write("| " + " | ".join(columns) + " |\n")
            f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
            for r in rows:
                if r["bucket"] != bucket:
                    continue
                f.write("| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |\n")
            f.write("\n")

    return md_path, csv_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a multimodal model's versatility across three tester buckets.",
    )
    parser.add_argument("config", help="Path to experiment JSON config.")
    parser.add_argument("--model", required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument(
        "--train-zips-manifest", default=None,
        help="Path to the <name>_train_zips.json sidecar (default: derived from config name).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--results-dir", default="tests/results",
        help="Output directory for Markdown + CSV (default: tests/results).",
    )
    args = parser.parse_args()

    with Path(args.config).open() as f:
        cfg = json.load(f)
    cfg.setdefault("test_ratio", 0.2)
    cfg.setdefault("seed", 42)
    name = cfg["name"]

    manifest_path = Path(
        args.train_zips_manifest
        or f"models/versatility_{name}_train_zips.json"
    )
    if not manifest_path.exists():
        print(
            f"Train-zips manifest not found: {manifest_path}. "
            f"Train first via src.train_versatility, or pass --train-zips-manifest.",
        )
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    train_zips = manifest["train_zips"]

    device = _get_device()
    print(f"Using device: {device}")

    model = _load_model(Path(args.model), device)

    all_rows: list[dict] = []

    # ---- Trained bucket ---------------------------------------------------
    print(f"\nEvaluating 'trained' on {len(train_zips)} zip(s) (test split)...")
    _, test_subset = split_multimodal_dataset(
        train_zips, test_ratio=cfg["test_ratio"], seed=cfg["seed"],
    )
    yt, yp, zs = _evaluate_subset(model, test_subset, device, args.batch_size)
    all_rows.extend(_rows_for_bucket("trained", yt, yp, zs))

    # ---- Similar & out_of_box buckets -------------------------------------
    for bucket_key in ("similar", "out_of_box"):
        patterns = cfg.get(bucket_key) or []
        if not patterns:
            print(f"Skipping '{bucket_key}' bucket — no patterns in config.")
            continue
        zips = _expand(patterns)
        if not zips:
            print(f"Skipping '{bucket_key}' bucket — no zips matched.")
            continue
        print(f"\nEvaluating '{bucket_key}' on {len(zips)} zip(s)...")
        ds = MultimodalGameDataset(zips)
        if len(ds) == 0:
            print(f"  Bucket '{bucket_key}' contained 0 samples — skipping.")
            ds.close()
            continue
        full_subset = Subset(ds, list(range(len(ds))))
        yt, yp, zs = _evaluate_subset(model, full_subset, device, args.batch_size)
        all_rows.extend(_rows_for_bucket(bucket_key, yt, yp, zs))
        ds.close()

    # ---- Write outputs ----------------------------------------------------
    md_path, csv_path = _write_outputs(
        Path(args.results_dir), name, all_rows, cfg, train_zips, Path(args.model),
    )
    print(f"\nWrote {md_path}")
    print(f"Wrote {csv_path}")

    # ---- Terminal summary -------------------------------------------------
    print("\nSummary (ALL rows per bucket):")
    for r in all_rows:
        if r["tester"] == "ALL":
            print(
                f"  {r['bucket']:<12} n={r['num_samples']:<5} "
                f"overall={_fmt(r['overall_acc'])}"
            )


if __name__ == "__main__":
    main()
