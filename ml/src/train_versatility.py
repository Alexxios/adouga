"""Config-driven trainer for the versatility-evaluation workflow.

Reads a JSON experiment config, expands the ``trained`` bucket's globs, and
calls :func:`src.train_multimodal.train_multimodal` on that subset. Saves the
resulting checkpoint to ``models/versatility_<name>.pth`` and a sidecar
``models/versatility_<name>_train_zips.json`` listing the exact (sorted) zip
paths used — which lets the evaluator regenerate the identical 80/20 split.

Usage::

    python -m src.train_versatility experiments/games_only_v1.json
    python -m src.train_versatility experiments/games_only_v1.json --epochs 10
"""

import argparse
import json
import logging
import sys
from glob import glob
from pathlib import Path

from src.train_multimodal import train_multimodal

logger = logging.getLogger(__name__)


def _expand(patterns: list[str]) -> list[str]:
    """Expand a list of paths/globs into a sorted, deduplicated list."""
    out: set[str] = set()
    for pat in patterns:
        matches = glob(pat)
        if not matches and Path(pat).exists():
            matches = [pat]
        if not matches:
            logger.warning("No matches for pattern: %s", pat)
        out.update(matches)
    return sorted(out)


def _load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    for key in ("name", "trained"):
        if key not in cfg:
            raise ValueError(f"Config missing required key: {key!r}")
    cfg.setdefault("test_ratio", 0.2)
    cfg.setdefault("seed", 42)
    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Train the multimodal classifier on a config-specified subset.",
    )
    parser.add_argument("config", help="Path to experiment JSON config.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--unfreeze-epoch", type=int, default=5)
    parser.add_argument(
        "--models-dir", default="models",
        help="Directory for checkpoint + sidecar (default: models)",
    )
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    train_zips = _expand(cfg["trained"])
    if not train_zips:
        print("No zips matched the 'trained' bucket — aborting.")
        sys.exit(1)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    name = cfg["name"]
    ckpt_path = models_dir / f"versatility_{name}.pth"
    sidecar_path = models_dir / f"versatility_{name}_train_zips.json"

    sidecar_path.write_text(json.dumps({
        "name": name,
        "train_zips": train_zips,
        "test_ratio": cfg["test_ratio"],
        "seed": cfg["seed"],
    }, indent=2))
    print(f"Wrote train-zip manifest to {sidecar_path}")

    print(f"Training '{name}' on {len(train_zips)} zip(s) → {ckpt_path}")
    train_multimodal(
        zip_paths=train_zips,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_epoch=args.unfreeze_epoch,
        test_ratio=cfg["test_ratio"],
        output_path=str(ckpt_path),
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()
