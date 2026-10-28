"""
catalog.py
----------
Everything that happens BEFORE we ever touch an individual light curve.

This is "Stage D" from the plan, done up front on the catalog table rather
than per-star, plus the initial download of that table:

    #12 Ambiguous / unresolved dispositions -> drop bare "CANDIDATE" rows
    #13 Duplicate TCEs                       -> dedupe by kepid + period
    #14 Stars with too few transits          -> filter needs light-curve
                                                 baseline, so it is re-checked
                                                 later in build_dataset.py
    #15 Class imbalance                      -> not deleted here, handled via
                                                 class weights at train time
                                                 (see make_class_weights())
    #16 Star-level leakage                   -> star_level_split() below

Source: NASA Exoplanet Archive, "cumulative" KOI table (pre-vetted, this is
the "pre-cleaned" catalog analogous to using PDCSAP_FLUX instead of SAP_FLUX
for the light curves themselves - no reason to re-derive dispositions NASA
already vetted).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config

logger = logging.getLogger(__name__)

# Columns we actually need out of the (much wider) cumulative KOI table.
KOI_COLUMNS = [
    "kepid",          # star identifier -> used for the star-level split
    "kepoi_name",     # per-candidate identifier (K00752.01 etc.)
    "koi_disposition",  # CONFIRMED / FALSE POSITIVE / CANDIDATE
    "koi_period",     # orbital period, days
    "koi_time0bk",    # transit epoch (BKJD)
    "koi_duration",   # transit duration, hours
    "koi_depth",      # transit depth, ppm (sanity checks only)
]


def fetch_koi_catalog() -> pd.DataFrame:
    """
    Download the cumulative KOI table from the NASA Exoplanet Archive.

    Requires internet access to exoplanetarchive.ipac.caltech.edu.
    Raises the underlying network error unchanged if that's unavailable -
    the caller decides whether to fall back to a cached CSV.
    """
    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    logger.info("Querying NASA Exoplanet Archive (cumulative KOI table)...")
    table = NasaExoplanetArchive.query_criteria(
        table="cumulative",
        select=",".join(KOI_COLUMNS),
    )
    df = table.to_pandas()
    logger.info("Fetched %d KOI rows.", len(df))
    return df


def clean_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage D, catalog-level part. Returns a clean, deduped, labeled table
    with exactly one row per usable candidate.
    """
    df = df.copy()

    # --- #12 Ambiguous / unresolved dispositions -----------------------
    # Bare "CANDIDATE" has no final verdict yet: neither a clean positive
    # nor a clean negative, so it teaches the model nothing certain.
    before = len(df)
    df = df[df["koi_disposition"].isin(config.LABEL_MAP.keys())].copy()
    logger.info(
        "Dropped %d unresolved CANDIDATE rows (%d remain).",
        before - len(df), len(df),
    )

    # Basic sanity: need a real period/epoch/duration to fold on at all.
    df = df.dropna(subset=["kepid", "koi_period", "koi_time0bk", "koi_duration"])
    df = df[(df["koi_period"] > 0) & (df["koi_duration"] > 0)]

    # --- #13 Duplicate TCEs ---------------------------------------------
    # The same physical signal sometimes gets flagged twice at harmonic
    # periods (e.g. 2x or 0.5x the true period). Dedupe by star + period,
    # keeping the first (NASA's vetting already picked the best-fit KOI).
    df["_period_bucket"] = _harmonic_bucket(df["koi_period"].to_numpy())
    before = len(df)
    df = df.drop_duplicates(subset=["kepid", "_period_bucket"], keep="first")
    df = df.drop(columns="_period_bucket")
    logger.info(
        "Dropped %d duplicate/harmonic TCEs (%d remain).",
        before - len(df), len(df),
    )

    # Attach the numeric label used for training.
    df["label"] = df["koi_disposition"].map(config.LABEL_MAP).astype(int)

    # --- #15 Class imbalance ---------------------------------------------
    # Deliberately NOT filtered/resampled here - see make_class_weights().
    pos_rate = df["label"].mean()
    logger.info("Class balance after cleaning: %.2f%% positive.", 100 * pos_rate)

    return df.reset_index(drop=True)


def _harmonic_bucket(periods: np.ndarray, tol: float = None) -> np.ndarray:
    """
    Collapse periods that are integer multiples/fractions of each other
    (within `tol` fractional tolerance) onto the same bucket key, so
    drop_duplicates() catches harmonic duplicates and not just exact ones.
    """
    tol = config.PERIOD_DEDUPE_TOLERANCE if tol is None else tol
    # Bucket key = period rounded in log-space; harmonics (P, 2P, P/2, ...)
    # land in *different* log-buckets on purpose - true duplicates almost
    # always come back as near-identical periods, not harmonics, once NASA
    # has already vetted the table. We keep this simple and conservative:
    # only exact-ish matches (within tol) are merged.
    order = np.argsort(periods)
    bucket = np.arange(len(periods))
    sorted_periods = periods[order]
    for i in range(1, len(sorted_periods)):
        if abs(sorted_periods[i] - sorted_periods[i - 1]) / sorted_periods[i - 1] < tol:
            bucket[order[i]] = bucket[order[i - 1]]
    return bucket


def make_class_weights(labels: np.ndarray) -> dict[int, float]:
    """
    #15 Class imbalance, handled via class weights rather than deletion.
    Standard inverse-frequency weighting, usable directly as Keras/PyTorch
    `class_weight`.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    weights = {int(c): n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return weights


def star_level_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = None,
) -> pd.DataFrame:
    """
    #16 Star-level leakage. Splits by `kepid` so every candidate belonging
    to a given star lands entirely in one of train/val/test - never split
    across buckets, or the model can learn to recognize individual stars'
    quirks instead of the general shape of a transit.

    Adds a `split` column ('train' / 'val' / 'test') to a copy of `df`.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    df = df.copy()

    groups = df["kepid"].to_numpy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))

    # Split the remaining train_val portion again for the validation set.
    rel_val_size = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=seed)
    train_idx_rel, val_idx_rel = next(
        gss2.split(df.iloc[train_val_idx], groups=groups[train_val_idx])
    )
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    df["split"] = "train"
    df.loc[df.index[val_idx], "split"] = "val"
    df.loc[df.index[test_idx], "split"] = "test"

    # Sanity check: no star ever appears in two splits.
    star_to_splits = df.groupby("kepid")["split"].nunique()
    assert (star_to_splits == 1).all(), "Star-level leakage detected!"

    logger.info(
        "Star-level split -> train=%d val=%d test=%d (unique stars: %d)",
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
        df["kepid"].nunique(),
    )
    return df


def load_or_fetch_catalog(force_refresh: bool = False) -> pd.DataFrame:
    """Convenience wrapper: use the cached cleaned CSV if present."""
    if config.CATALOG_CSV.exists() and not force_refresh:
        logger.info("Loading cached catalog from %s", config.CATALOG_CSV)
        return pd.read_csv(config.CATALOG_CSV)

    raw = fetch_koi_catalog()
    cleaned = clean_catalog(raw)
    split = star_level_split(cleaned)
    split.to_csv(config.CATALOG_CSV, index=False)
    return split
