# Kepler transit-detection data pipeline

Downloads Kepler light curves + labels straight from **MAST** and the
**NASA Exoplanet Archive** and cleans them into fixed-size global/local
views ready to train a transit-detection CNN (the AstroNet-style approach:
2001-point "whole orbit" view + 201-point "zoomed on the transit" view).



## Setup

```bash
cd exoplanet_pipeline
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Run

```bash
# Recommended first: a quick pass over a small number of stars to make sure
# everything downloads and runs cleanly end to end before committing to the
# full multi-GB run.
python run_pipeline.py --max-targets 50

# Full run: every cleaned/labeled KOI host star, capped at
# config.MAX_DOWNLOAD_BYTES (12 GB by default) of raw FITS files.
python run_pipeline.py
```

This is safe to stop and re-run: the KOI catalog, each star's cleaned light
curve, and the downloaded FITS files are all cached to disk, so re-running
picks up roughly where it left off instead of re-downloading everything.

Output: `data/transit_dataset.npz` containing `global_view` (N×2001),
`local_view` (N×201), `label` (N,), `kepid` (N,), and `split` (N,, one of
`"train"/"val"/"test"`), plus `data/class_weights.json` for handling class
imbalance in your loss function, and `data/star_level_split.csv` recording
which stars ended up in which split.

## What each file does

| File | Pipeline stage |
|---|---|
| `config.py` | every tunable constant, in one place |
| `catalog.py` | downloads the KOI "cumulative" table (NASA Exoplanet Archive) and does the **Stage D** label/dataset cleaning: drops ambiguous `CANDIDATE` rows (#12), dedupes harmonic/duplicate TCEs (#13), computes class weights instead of deleting to fix imbalance (#15), and does the star-level train/val/test split so no star leaks across splits (#16) |
| `lightcurves.py` | downloads each star's light curve from MAST via `lightkurve` and applies **Stage A** (drop NaNs, quality-flagged cadences, extreme outliers, non-positive flux) and **Stage B** items #5/#7 (uses `PDCSAP_FLUX`, stitches quarters together) |
| `views.py` | **Stage B** #6 (flattens stellar variability while protecting the transit) and **Stage C** (phase-folds, bins into fixed 2001/201-point views without zero-filling gaps, median-normalizes); also re-checks **Stage D** #14 exactly, now that a real observed time baseline exists |
| `dataset.py` | orchestrates the above across every star/candidate and saves the final `.npz` |
| `run_pipeline.py` | single entry point (`catalog` → `dataset`) |

## Sizing

Each star's full-mission long-cadence `PDCSAP_FLUX` light curve is roughly
1–3 MB once downloaded. The cleaned/labeled KOI catalog has on the order of
several thousand host stars, so a full run typically lands in the 10–12 GB
range you mentioned — `config.MAX_DOWNLOAD_BYTES` enforces a hard stop there
regardless. Use `--max-targets` for a smaller run (e.g. `--max-targets 300`
is roughly 1–2 GB) while you're still debugging the model itself.

## Adjusting the cleaning behavior

Everything referenced above by stage/item number is a knob in `config.py` —
e.g. `OUTLIER_SIGMA`, `FLATTEN_WINDOW_LENGTH`, `MIN_TRANSITS_REQUIRED`,
`GLOBAL_VIEW_BINS`/`LOCAL_VIEW_BINS`, `QUALITY_BITMASK` (`'default'` vs.
`'hard'`) — change the constant, re-run, no need to touch the pipeline logic.
