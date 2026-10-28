"""
dataset.py
----------
Top-level orchestration. For every candidate in the cleaned, star-level-split
KOI catalog (catalog.py), builds a (global_view, local_view, label) example
using lightcurves.py + views.py, respecting the download budget, and saves
one combined .npz plus a class-weights file.

Usage:
    python dataset.py                     # full run (up to MAX_DOWNLOAD_BYTES)
    python dataset.py --max-targets 300   # quick pass over 300 stars only
    python dataset.py --fresh             # ignore any existing checkpoint, start over
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

import config
import catalog
import lightcurves
import views

logger = logging.getLogger(__name__)

PROCESSED_KEPIDS_JSON = config.DATA_DIR / "processed_kepids.json"


def load_checkpoint() -> tuple[list[dict], set[int]]:
    """
    Reload whatever a previous (possibly interrupted) run already saved,
    so a crash never means starting over from zero.
    """
    examples: list[dict] = []

    if config.DATASET_NPZ.exists():
        data = np.load(config.DATASET_NPZ, allow_pickle=False)

        # WHY each array is read ONCE into its own variable, BEFORE the loop:
        # an .npz file is a compressed archive, and every time you write
        # data["global_view"] it decompresses and reads that WHOLE array
        # from disk again. The old version wrote data["global_view"][i]
        # inside the loop, so for 3,835 examples it decompressed the full
        # (3835 x 2001) array 3,835 times over, which ran out of memory.
        # Reading each array once and indexing the in-memory copy is
        # identical in result but thousands of times cheaper.
        global_views = data["global_view"]
        local_views = data["local_view"]
        labels = data["label"]
        kepids = data["kepid"]
        splits = data["split"]
        # older checkpoint files were saved before kepoi_name existed
        names = data["kepoi_name"] if "kepoi_name" in data.files else None

        for i in range(len(labels)):
            examples.append({
                "kepid": int(kepids[i]),
                "kepoi_name": str(names[i]) if names is not None else "",
                "label": int(labels[i]),
                "split": str(splits[i]),
                "global_view": global_views[i],
                "local_view": local_views[i],
            })
        logger.info("Resumed %d examples from existing %s", len(examples), config.DATASET_NPZ)

    processed_kepids: set[int] = set()
    if PROCESSED_KEPIDS_JSON.exists():
        with open(PROCESSED_KEPIDS_JSON) as f:
            processed_kepids = set(json.load(f))
        logger.info("Resuming: %d stars already attempted, will be skipped.", len(processed_kepids))

    return examples, processed_kepids


def save_checkpoint(examples: list[dict], processed_kepids: set[int]) -> None:
    """Lightweight periodic save - cheap enough to call every N stars."""
    if examples:
        np.savez_compressed(
            config.DATASET_NPZ,
            global_view=np.stack([e["global_view"] for e in examples]),
            local_view=np.stack([e["local_view"] for e in examples]),
            label=np.array([e["label"] for e in examples], dtype=np.int64),
            kepid=np.array([e["kepid"] for e in examples], dtype=np.int64),
            kepoi_name=np.array([e["kepoi_name"] for e in examples]),
            split=np.array([e["split"] for e in examples]),
        )
    with open(PROCESSED_KEPIDS_JSON, "w") as f:
        json.dump(sorted(processed_kepids), f)


def build_examples(df: pd.DataFrame, resume: bool = True,
                    checkpoint_every: int = 25) -> list[dict]:
    if resume:
        examples, processed_kepids = load_checkpoint()
    else:
        examples, processed_kepids = [], set()

    stars = df["kepid"].unique()
    remaining = [k for k in stars if int(k) not in processed_kepids]
    skipped = len(stars) - len(remaining)
    if skipped:
        logger.info("Skipping %d already-processed stars from a previous run.", skipped)

    for i, kepid in enumerate(remaining, 1):
        if lightcurves.download_budget_remaining() <= 0:
            logger.warning(
                "Hit the %.1f GB download budget after %d/%d remaining stars - stopping "
                "new downloads (already-cached stars are unaffected).",
                config.MAX_DOWNLOAD_BYTES / 1024 ** 3, i - 1, len(remaining),
            )
            break

        lc = lightcurves.get_clean_lightcurve(int(kepid))
        if lc is not None:
            for _, koi in df[df["kepid"] == kepid].iterrows():
                period, t0, dur = koi["koi_period"], koi["koi_time0bk"], koi["koi_duration"]

                # Stage D #14, exact re-check now that we have the real light curve.
                if views.count_observed_transits(lc, period, t0) < config.MIN_TRANSITS_REQUIRED:
                    continue

                # WHY this try/except: a small number of pathological
                # period/duration combinations produce a transit mask
                # that covers (near-)100% of the light curve, leaving
                # lightkurve's internal flatten() with ZERO out-of-
                # transit points to build a trend from. That throws a
                # raw ValueError deep inside scipy's interp1d - one bad
                # candidate out of thousands should never be allowed to
                # kill an entire multi-hour run. Skip it, log exactly
                # which candidate and why, and keep going.
                try:
                    flat = views.flatten_for_transit(lc, period, t0, dur)
                    if len(flat.time) == 0:
                        continue

                    gv = views.make_global_view(flat, period, t0)
                    lv = views.make_local_view(flat, period, t0, dur)
                    if gv is None or lv is None:
                        continue

                    examples.append({
                        "kepid": int(kepid),
                        "kepoi_name": koi["kepoi_name"],
                        "label": int(koi["label"]),
                        "split": koi["split"],
                        "global_view": gv.astype(np.float32),
                        "local_view": lv.astype(np.float32),
                    })
                except Exception as e:
                    logger.warning(
                        "  Skipping KIC %s / %s - flatten/view generation failed: %s",
                        kepid, koi["kepoi_name"], e,
                    )
                    continue

        # Mark as processed even on failure/zero examples - retrying a
        # star that already failed just wastes time and re-triggers the
        # same network calls.
        processed_kepids.add(int(kepid))

        if i % checkpoint_every == 0:
            save_checkpoint(examples, processed_kepids)
            logger.info("  checkpoint: %d/%d remaining stars processed, %d examples so far",
                        i, len(remaining), len(examples))

    # Final save so the last partial batch isn't lost either.
    save_checkpoint(examples, processed_kepids)
    return examples


def save_examples(examples: list[dict]) -> None:
    if not examples:
        logger.warning("No examples were built - nothing to save.")
        return

    # Arrays are already saved incrementally by save_checkpoint(); this
    # only handles the summary outputs that make sense once the run is done.
    split_summary = pd.DataFrame(
        {"kepid": [e["kepid"] for e in examples], "split": [e["split"] for e in examples]}
    ).drop_duplicates()
    split_summary.to_csv(config.SPLIT_CSV, index=False)

    for split_name in ("train", "val", "test"):
        rows = [e for e in examples if e["split"] == split_name]
        pos = sum(r["label"] for r in rows)
        logger.info("  %-5s: %5d examples (%d positive / %d negative)",
                    split_name, len(rows), pos, len(rows) - pos)

    train_labels = np.array([e["label"] for e in examples if e["split"] == "train"])
    if len(train_labels):
        weights = catalog.make_class_weights(train_labels)
        weights_path = config.DATA_DIR / "class_weights.json"
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)
        logger.info("Class weights (train split) -> %s : %s", weights_path, weights)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-targets", type=int, default=config.MAX_TARGETS,
                         help="Only process this many stars (for a quick test run).")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing checkpoint and start over from scratch.")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                         help="Save progress to disk every N stars (default: 25).")
    args = parser.parse_args()

    df = catalog.load_or_fetch_catalog()
    if args.max_targets:
        keep_stars = df["kepid"].drop_duplicates().head(args.max_targets)
        df = df[df["kepid"].isin(keep_stars)]
        logger.info("Limiting to %d stars (--max-targets).", args.max_targets)

    examples = build_examples(df, resume=not args.fresh, checkpoint_every=args.checkpoint_every)
    logger.info("Built %d total examples from %d candidate stars.", len(examples), df["kepid"].nunique())
    save_examples(examples)


if __name__ == "__main__":
    main()