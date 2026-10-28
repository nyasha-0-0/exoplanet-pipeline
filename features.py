"""
features.py — Phase 8: Hand-Crafted Feature Table

WHY this file exists separately from views.py:
views.py produces RAW ARRAYS (2001/201-point shape sequences) for the CNN —
the CNN learns what matters about that shape on its own, with no human
telling it "depth matters" or "duration matters."

This file instead computes a small table of SIX HUMAN-DEFINED NUMBERS per
candidate. These numbers power two very different downstream consumers:
  1. Phase 9's baseline model (a classical ML model with no notion of
     "shape" at all — it only sees these 6 numbers)
  2. Phase 12's physics validation layer (the same 6 numbers, reused as
     literal astrophysical sanity-checks, e.g. "is the odd/even depth
     difference small enough to rule out an eclipsing binary?")

HOW: this script does NOT re-download anything. It reuses the exact same
cached, cleaned light curve that lightcurves.py already saved to disk
during the dataset.py run, and the exact same flattening/folding logic
already written in views.py — so a star that already has a cached cleaned
FITS file costs zero new network calls here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
import lightcurves
import views

logger = logging.getLogger(__name__)

FEATURES_CSV = config.DATA_DIR / "features.csv"


def _out_of_transit_mask(time: np.ndarray, period: float, t0: float,
                          duration_hours: float) -> np.ndarray:
    """
    True for cadences NOT near any transit — i.e. the "quiet" baseline
    flux. Reuses views.transit_mask() and just inverts it, so the
    definition of "in transit" is identical everywhere in the codebase —
    never redefine the same window twice in two places.
    """
    return ~views.transit_mask(time, period, t0, duration_hours)


def compute_depth_and_snr(flat_flux: np.ndarray, in_transit: np.ndarray) -> tuple[float, float]:
    """
    WHY recompute depth from the actual cleaned, flattened light curve
    instead of just trusting koi_depth from the catalog: koi_depth is
    NASA's own pipeline's measurement, taken on THEIR cleaning of the
    data, not ours. Recomputing it on our own Stage A/B output is both
    a sanity check (do our numbers roughly agree with NASA's?) and
    gives us a number computed consistently with everything else here.

    depth: how far the in-transit median flux sits BELOW the baseline
    (1.0, since make_global_view()/flatten already center flux near 1).
    SNR: depth divided by the scatter (std) of the OUT-of-transit flux —
    a simple, standard "signal over noise" ratio. A deep dip on a very
    noisy star (large std) is less trustworthy than the same depth on a
    quiet star, and this one number captures that trade-off.
    """
    out_flux = flat_flux[~in_transit]
    in_flux = flat_flux[in_transit]

    if len(in_flux) == 0 or len(out_flux) == 0:
        return np.nan, np.nan

    baseline = np.nanmedian(out_flux)
    depth = baseline - np.nanmedian(in_flux)  # positive number = a real dip
    noise = np.nanstd(out_flux)
    snr = depth / noise if noise > 0 else np.nan
    return float(depth), float(snr)


def compute_odd_even_depth_diff(time: np.ndarray, flat_flux: np.ndarray,
                                 period: float, t0: float,
                                 duration_hours: float) -> float:
    """
    WHY this matters: a background eclipsing binary at HALF the detected
    period produces transits that alternate between two slightly
    different depths (the two stars have different sizes/brightness) —
    but a real planet's transits should all look the same. A LARGE
    odd/even difference is a classic false-positive red flag; a small
    one is consistent with a real planet.

    HOW: figure out which numbered transit cycle each cadence belongs to
    (floor((time - t0) / period)), split cadences into transits whose
    cycle number is even vs odd, and compare each group's median depth.
    """
    in_transit = views.transit_mask(time, period, t0, duration_hours)
    if not in_transit.any():
        return np.nan

    cycle_number = np.floor((time[in_transit] - t0) / period + 0.5).astype(int)
    is_even_cycle = (cycle_number % 2 == 0)

    out_flux = flat_flux[~in_transit]
    baseline = np.nanmedian(out_flux) if len(out_flux) else np.nan
    in_flux = flat_flux[in_transit]

    even_depth = baseline - np.nanmedian(in_flux[is_even_cycle]) if is_even_cycle.any() else np.nan
    odd_depth = baseline - np.nanmedian(in_flux[~is_even_cycle]) if (~is_even_cycle).any() else np.nan

    if np.isnan(even_depth) or np.isnan(odd_depth):
        return np.nan
    return float(abs(even_depth - odd_depth))


def compute_secondary_eclipse_depth(time: np.ndarray, flat_flux: np.ndarray,
                                     period: float, t0: float,
                                     duration_hours: float) -> float:
    """
    WHY this matters: a self-luminous companion (i.e. a second star, not
    a planet) produces its OWN small dip exactly half an orbit away from
    the main transit (phase 0.5), because it briefly goes behind the
    primary star too. A real planet is not self-luminous and should show
    no such dip. A large secondary-eclipse depth is a false-positive
    giveaway; near-zero is consistent with a real planet.

    HOW: identical transit_mask() logic as the primary transit, but
    phase-shifted by half a period (t0 + period/2) before checking which
    cadences are "in-transit."
    """
    secondary_t0 = t0 + period / 2.0
    in_secondary = views.transit_mask(time, period, secondary_t0, duration_hours)
    if not in_secondary.any():
        return np.nan

    out_flux = flat_flux[~in_secondary]
    baseline = np.nanmedian(out_flux) if len(out_flux) else np.nan
    sec_flux = np.nanmedian(flat_flux[in_secondary])

    if np.isnan(baseline):
        return np.nan
    return float(baseline - sec_flux)  # positive = a real secondary dip


def build_feature_row(lc, koi: pd.Series) -> dict | None:
    """
    Computes all 6 hand-crafted features for ONE candidate, given its
    star's already-cleaned light curve. Returns None if there isn't
    enough usable data to compute a trustworthy row (mirrors the same
    "skip, don't crash" philosophy as dataset.py's build_examples()).
    """
    period = koi["koi_period"]
    t0 = koi["koi_time0bk"]
    duration_hours = koi["koi_duration"]

    # Wrap flattening in try-except to catch empty-array interpolation crashes
    try:
        flat = views.flatten_for_transit(lc, period, t0, duration_hours)
        if flat is None or len(flat.time) == 0:
            return None
    except Exception:
        return None

    time = flat.time.value
    flux = flat.flux.value
    in_transit = views.transit_mask(time, period, t0, duration_hours)

    if not in_transit.any() or in_transit.all():
        # WHY: no transits in this star's observed baseline, or (degenerate
        # edge case) the "transit" window somehow covers the ENTIRE light
        # curve — either way there's no valid in/out-of-transit comparison
        # to make, so skip rather than return garbage.
        return None

    depth, snr = compute_depth_and_snr(flux, in_transit)
    odd_even_diff = compute_odd_even_depth_diff(time, flux, period, t0, duration_hours)
    secondary_depth = compute_secondary_eclipse_depth(time, flux, period, t0, duration_hours)

    return {
        "kepid": int(koi["kepid"]),
        "kepoi_name": koi["kepoi_name"],
        "label": int(koi["label"]),
        "split": koi["split"],
        "depth": depth,
        "duration_hours": float(duration_hours),
        "period_days": float(period),
        "snr": snr,
        "odd_even_depth_diff": odd_even_diff,
        "secondary_eclipse_depth": secondary_depth,
    }


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loops star-by-star (same pattern as dataset.py's build_examples()) so
    each star's cached light curve is loaded ONCE and reused for all of
    that star's candidates, rather than re-loading per candidate.
    """
    rows = []
    stars = df["kepid"].unique()

    for i, kepid in enumerate(stars, 1):
        # WHY this call is cheap here even for thousands of stars: every
        # star in df was already downloaded and cached during dataset.py's
        # run. get_clean_lightcurve() checks the local cache FIRST (see
        # lightcurves.py) and only hits the network if that cache is
        # missing — so this is a disk-read, not a re-download, for any
        # star dataset.py already finished.
        lc = lightcurves.get_clean_lightcurve(int(kepid))
        if lc is None:
            continue

        for _, koi in df[df["kepid"] == kepid].iterrows():
            row = build_feature_row(lc, koi)
            if row is not None:
                rows.append(row)

        if i % 100 == 0:
            logger.info("  %d/%d stars processed, %d feature rows so far", i, len(stars), len(rows))

    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = pd.read_csv(config.CATALOG_CSV)
    logger.info("Loaded %d candidates from %s", len(df), config.CATALOG_CSV)

    features_df = build_feature_table(df)
    logger.info("Built %d feature rows (of %d candidates attempted).", len(features_df), len(df))

    # WHY report NaN counts per column before saving: a hand-crafted
    # feature that's NaN for a large fraction of rows is a silent data
    # quality problem Phase 9's baseline model would otherwise inherit
    # without warning.
    for col in ["depth", "snr", "odd_even_depth_diff", "secondary_eclipse_depth"]:
        n_nan = features_df[col].isna().sum()
        if n_nan:
            logger.warning("  %s: %d/%d rows are NaN (%.1f%%)", col, n_nan, len(features_df),
                            100 * n_nan / len(features_df) if len(features_df) else 0)

    features_df.to_csv(FEATURES_CSV, index=False)
    logger.info("Saved feature table -> %s", FEATURES_CSV)

    for split_name in ("train", "val", "test"):
        n = (features_df["split"] == split_name).sum()
        pos = features_df.loc[features_df["split"] == split_name, "label"].sum()
        logger.info("  %-5s: %5d rows (%d positive / %d negative)", split_name, n, pos, n - pos)


if __name__ == "__main__":
    main()