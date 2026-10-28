"""
config.py
---------
All the tunable knobs for the pipeline live here so the rest of the code
never hard-codes a magic number. Matches the "Stage A/B/C/D" plan:

    Stage A - bad measurements      (remove rows entirely)
    Stage B - unwanted signals      (keep rows, remove the trend)
    Stage C - structural problems   (fix the shape of the data)
    Stage D - label / dataset level cleaning (not per-light-curve)
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "lightkurve_cache"     # raw FITS files land here
DATA_DIR = PROJECT_ROOT / "data"                  # intermediate + final .npz/.csv
CATALOG_CSV = DATA_DIR / "koi_catalog_clean.csv"   # Stage D output
DATASET_NPZ = DATA_DIR / "transit_dataset.npz"     # final model-ready arrays
SPLIT_CSV = DATA_DIR / "star_level_split.csv"

for d in (CACHE_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Stage A - bad measurements (row-level, always safe to drop)
# ----------------------------------------------------------------------
QUALITY_BITMASK = "default"     # lightkurve: 'default' | 'hard' | 'hardest'
OUTLIER_SIGMA = 5.0             # lc.remove_outliers(sigma=OUTLIER_SIGMA)

# ----------------------------------------------------------------------
# Stage B - unwanted signals (detrending)
# ----------------------------------------------------------------------
FLATTEN_WINDOW_LENGTH = 401     # lc.flatten(window_length=...), must be odd
FLATTEN_POLYORDER = 2
TRANSIT_MASK_PAD_DURATIONS = 1.5  # how many transit-durations either side of
                                   # a transit to protect from the flattener

CLEANED_LC_DIR = DATA_DIR / "cleaned_lightcurves"  # one stitched+Stage-A-cleaned
CLEANED_LC_DIR.mkdir(parents=True, exist_ok=True)  # FITS file per star, cached

# ----------------------------------------------------------------------
# Stage C - structural problems (fixed-size views for the CNN)
# ----------------------------------------------------------------------
GLOBAL_VIEW_BINS = 2001         # zoomed-out, whole orbital period
LOCAL_VIEW_BINS = 201           # zoomed-in, just the transit itself
LOCAL_VIEW_NUM_DURATIONS = 4.0  # local view spans +/- this many transit durations

# ----------------------------------------------------------------------
# Stage D - label / dataset level cleaning
# ----------------------------------------------------------------------
# NASA Exoplanet Archive "koi_disposition" values we accept as a definite
# label. "CANDIDATE" with no final verdict is ambiguous and is dropped.
LABEL_MAP = {
    "CONFIRMED": 1,
    "FALSE POSITIVE": 0,
}
MIN_TRANSITS_REQUIRED = 3       # need >=3 transits in the baseline to fold usefully
PERIOD_DEDUPE_TOLERANCE = 0.01  # fractional tolerance when matching harmonics

# ----------------------------------------------------------------------
# Download budget (this is what "download up to 10-12 GB" maps to in code)
# ----------------------------------------------------------------------
MAX_DOWNLOAD_BYTES = 12 * 1024 ** 3   # hard stop at 12 GB of raw FITS files
MISSION = "Kepler"
FLUX_COLUMN = "pdcsap_flux"           # PDCSAP_FLUX, not SAP_FLUX - see Stage B/#5

# None = process every cleaned/split KOI host star (the full ~10-12GB run).
# Set an int (e.g. 300) for a fast first pass while you debug the pipeline.
MAX_TARGETS = None

# Reproducibility
RANDOM_SEED = 42
