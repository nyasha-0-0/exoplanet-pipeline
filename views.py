"""
views.py
--------
Per-(star, candidate) processing: turns one star's cleaned light curve (from
lightcurves.py) plus one candidate's ephemeris (period / epoch / duration)
into the fixed-size global/local views a CNN can actually consume.

    Stage B #6  Stellar variability -> lc.flatten(window_length=..., mask=...)
                The transit itself is masked out first so the flattener
                doesn't smooth away the very dip we're trying to detect.

    Stage C     Structural problems
        #8  Data gaps                 -> interpolate, never zero-fill
        #9  Empty bins after folding  -> same: interpolate, don't zero-fill
        #10 Inconsistent array length -> fixed GLOBAL_VIEW_BINS / LOCAL_VIEW_BINS
        #11 Unnormalized flux scale   -> median-normalize each view

    Stage D #14 (exact re-check) Stars with too few transits: catalog.py can
    only approximate this from the mission-wide baseline before any light
    curve has been downloaded; here we check it exactly against the star's
    real observed time span (gaps included).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import lightkurve as lk

import config


def transit_mask(time: np.ndarray, period: float, t0: float, duration_hours: float) -> np.ndarray:
    """True for cadences inside/near a transit - protects them from flatten()."""
    duration_days = duration_hours / 24.0
    half_width = config.TRANSIT_MASK_PAD_DURATIONS * duration_days / 2.0
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase) < half_width


def count_observed_transits(lc: "lk.LightCurve", period: float, t0: float) -> int:
    """Stage D #14, exact version: how many transits actually fall inside
    this star's real observed time span, not the mission-wide estimate
    catalog.py uses before any data has been downloaded."""
    t_start, t_end = float(lc.time.value.min()), float(lc.time.value.max())
    first_n = np.ceil((t_start - t0) / period)
    last_n = np.floor((t_end - t0) / period)
    return int(max(0, last_n - first_n + 1))


def flatten_for_transit(lc: "lk.LightCurve", period: float, t0: float,
                         duration_hours: float) -> "lk.LightCurve":
    """Stage B #6."""
    mask = transit_mask(lc.time.value, period, t0, duration_hours)
    return lc.flatten(
        window_length=config.FLATTEN_WINDOW_LENGTH,
        polyorder=config.FLATTEN_POLYORDER,
        mask=mask,
    )


def _median_bin(x: np.ndarray, y: np.ndarray, num_bins: int,
                 x_min: float, x_max: float) -> Optional[np.ndarray]:
    edges = np.linspace(x_min, x_max, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(x, edges) - 1, 0, num_bins - 1)

    binned = np.full(num_bins, np.nan)
    for b in range(num_bins):
        vals = y[idx == b]
        if len(vals):
            binned[b] = np.median(vals)

    empty = np.isnan(binned)
    if empty.all():
        return None
    
    # FIX: If more than 20% of the local window is missing data gaps,
    # reject it instead of drawing a fake interpolated flat bottom!
    if empty.sum() > 0.2 * num_bins:
        return None

    if empty.any():
        binned[empty] = np.interp(centers[empty], centers[~empty], binned[~empty])
        
    return binned


def _normalize_view(view: np.ndarray) -> np.ndarray:
    """Stage C #11: median-normalize so different stars' absolute
    brightness scales don't matter to the CNN."""
    baseline = np.nanmedian(view)
    return (view / baseline - 1.0) if baseline else view


def make_global_view(lc: "lk.LightCurve", period: float, t0: float) -> Optional[np.ndarray]:
    """Stage C #10: fixed-length view of the *entire* folded orbit, for the
    CNN to see the overall shape."""
    folded = lc.fold(period=period, epoch_time=t0)
    binned = _median_bin(
        folded.time.value, folded.flux.value,
        config.GLOBAL_VIEW_BINS, -period / 2, period / 2,
    )
    return None if binned is None else _normalize_view(binned)


def make_local_view(lc: "lk.LightCurve", period: float, t0: float,
                     duration_hours: float) -> Optional[np.ndarray]:
    duration_days = duration_hours / 24.0
    half_window = config.LOCAL_VIEW_NUM_DURATIONS * duration_days / 2.0

    folded = lc.fold(period=period, epoch_time=t0)
    phase, flux = folded.time.value, folded.flux.value

    # FIX: Check a small window around phase 0.0 to re-center the dip if the catalog t0 is slightly off
    center_mask = np.abs(phase) < (duration_days * 0.5)
    if center_mask.any() and not np.isnan(flux[center_mask]).all():
        # Find the exact local minimum in the transit region
        local_argmin = np.nanargmin(flux[center_mask])
        phase_subset = phase[center_mask]
        t0_shift = phase_subset[local_argmin]
        
        # Re-fold with the corrected, perfectly centered epoch
        folded = lc.fold(period=period, epoch_time=t0 + t0_shift)
        phase, flux = folded.time.value, folded.flux.value

    keep = np.abs(phase) < half_window * 2  
    binned = _median_bin(phase[keep], flux[keep], config.LOCAL_VIEW_BINS, -half_window, half_window)
    return None if binned is None else _normalize_view(binned)
