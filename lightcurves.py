"""
lightcurves.py — Stages A & B (Per-Star Cleaning)

WHY this file exists separately from views.py:
Stage A/B cleaning happens ONCE per star (download, stitch, remove bad
cadences). Stage C (folding, binning) happens once per CANDIDATE — a
single star can have multiple KOIs. Splitting them avoids re-downloading
the same star's data for every candidate it hosts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import lightkurve as lk
from astroquery.mast import Observations

import config

log = logging.getLogger(__name__)

# WHY: MAST's default 600s timeout is too tight under load; give it more
# room before we fall back to retry/backoff.
Observations.TIMEOUT = 1200  # seconds (20 min)

# Retry policy for transient MAST timeouts
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # seconds: 2, 4, 8...


def _dir_size_bytes(path: Path) -> int:
    """Total size in bytes of everything under `path`."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def download_budget_remaining() -> int:
    """
    Bytes remaining before MAX_DOWNLOAD_BYTES is hit.

    WHY: prevents an unattended full run from silently filling the disk.
    Cached stars are exempt (see get_clean_lightcurve) — only NEW
    downloads are throttled by this check.
    """
    used = _dir_size_bytes(config.CACHE_DIR)
    return max(0, config.MAX_DOWNLOAD_BYTES - used)


def cleaned_lightcurve_path(kepid: int) -> Path:
    """Cache path for a star's Stage-A-cleaned, stitched light curve."""
    return config.CLEANED_LC_DIR / f"kic{kepid}.fits"


def get_clean_lightcurve(kepid: int):
    """
    Download (or load cached), stitch, and Stage-A clean one star's
    full Kepler mission light curve.

    Returns a lightkurve LightCurve object, or None if:
      - no data exists for this star, or
      - all retries were exhausted on a persistent MAST timeout.

    WHY return None instead of raising: dataset.py loops over hundreds
    of stars — one bad star should not kill a multi-hour run. The
    caller decides whether to skip and continue.
    """
    cache_path = cleaned_lightcurve_path(kepid)

    # --- Layer 2 cache: already-cleaned FITS file exists ---
    # WHY this check ignores the download budget entirely: a cached
    # star costs zero new bytes, so throttling it would be pointless
    # and would make re-runs slower for no reason.
    if cache_path.exists():
        try:
            return lk.io.read(str(cache_path))
        except Exception as e:
            log.warning("Cached file for KIC %s unreadable (%s); re-downloading.", kepid, e)
            cache_path.unlink(missing_ok=True)

    # --- Budget check (new downloads only) ---
    if download_budget_remaining() <= 0:
        log.warning("Download budget exhausted; skipping KIC %s.", kepid)
        return None

    # --- Layer 1: search + download with retry/backoff on timeout ---
    search = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            search = lk.search_lightcurve(
                f"KIC {kepid}",
                mission=config.MISSION,
                author="Kepler",
                cadence="long",
            )
            break  # success — exit retry loop
        except TimeoutError:
            wait = BACKOFF_FACTOR ** attempt
            log.warning(
                "MAST timeout searching KIC %s (attempt %d/%d). Retrying in %ds...",
                kepid, attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        except Exception as e:
            log.error("Unexpected error searching KIC %s: %s", kepid, e)
            return None
    else:
        # WHY: loop exhausted without a `break` — every retry timed out.
        # Log and skip rather than crashing the whole --max-targets run.
        log.error("MAST timeout on KIC %s after %d attempts. Skipping.", kepid, MAX_RETRIES)
        return None

    if search is None or len(search) == 0:
        log.info("No Kepler long-cadence data found for KIC %s.", kepid)
        return None

    # --- Download all quarters, same retry treatment ---
    collection = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            collection = search.download_all(
                quality_bitmask=config.QUALITY_BITMASK,
                flux_column=config.FLUX_COLUMN,  # PDCSAP_FLUX — Stage B #5
            )
            break
        except TimeoutError:
            wait = BACKOFF_FACTOR ** attempt
            log.warning(
                "MAST timeout downloading KIC %s (attempt %d/%d). Retrying in %ds...",
                kepid, attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        except Exception as e:
            log.error("Unexpected error downloading KIC %s: %s", kepid, e)
            return None
    else:
        log.error("MAST timeout downloading KIC %s after %d attempts. Skipping.", kepid, MAX_RETRIES)
        return None

    if collection is None or len(collection) == 0:
        return None

    # --- Stage B #7: stitch quarters together (normalizes per quarter) ---
    lc = collection.stitch()

    # --- Stage A cleaning ---
    lc = lc.remove_nans()                 # #1 — NaN flux
    lc = lc[lc.flux.value > 0]            # #4 — zero/negative flux (physically impossible)

    # #3 — extreme outliers, ASYMMETRIC clipping.
    # WHY asymmetric: a real transit IS a downward (negative) outlier.
    # Symmetric 5-sigma clipping on both sides would delete genuine
    # transit dips along with cosmic-ray noise. Clip the upper side
    # tightly (sigma_upper=5 — noise spikes are almost always positive
    # glitches) but leave far more room on the lower side
    # (sigma_lower=20) so a deep, real transit survives.
    lc = lc.remove_outliers(sigma_lower=20.0, sigma_upper=5.0)

    # --- Cache the cleaned result (Layer 2) ---
    config.CLEANED_LC_DIR.mkdir(parents=True, exist_ok=True)
    lc.to_fits(str(cache_path), overwrite=True)

    return lc