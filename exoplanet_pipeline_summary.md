# Comprehensive Summary — `exoplanet_pipeline`

> **Project path:** `C:\Users\nyash\PROJECT\EXOSCOPE\exoplanet_pipeline`

---

## 1. Executive Summary

### Primary Purpose
`exoplanet_pipeline` is a **self-contained Python data pipeline** that downloads real Kepler telescope observations from two primary astronomical sources — **MAST** (raw light curves) and the **NASA Exoplanet Archive** (labels) — and converts them into clean, fixed-size numerical arrays ready to train a **transit-detection Convolutional Neural Network (CNN)**.

### Main Problem / Objective
Detecting exoplanets from stellar light curves is a binary classification task: does the periodic dimming of a star's brightness represent a real planetary transit (`CONFIRMED = 1`) or a false alarm (`FALSE POSITIVE = 0`)? Before any model can be trained, the raw telescope data must be extensively cleaned, structured, and labeled. This pipeline automates that entire process.

### Key Outcomes / What It Produces
| Output File | Contents |
|---|---|
| `data/transit_dataset.npz` | Arrays: `global_view` (N×2001), `local_view` (N×201), `label` (N,), `kepid` (N,), `split` (N,) |
| `data/class_weights.json` | Inverse-frequency weights for handling class imbalance in the loss function |
| `data/star_level_split.csv` | Which Kepler star ID (`kepid`) landed in which split (train/val/test) |
| `data/koi_catalog_clean.csv` | The cleaned, labeled, split KOI catalog table |

### Most Important Conclusions
- The pipeline follows a **four-stage cleaning plan (Stages A→D)** covering bad measurements, unwanted signals, structural shape problems, and label/dataset-level issues.
- It deliberately avoids the pre-packaged Kaggle datasets, sourcing data directly from NASA/MAST to maintain full control over cleaning decisions.
- The pipeline is **idempotent and safe to interrupt**: every expensive download and cleaning step is cached to disk, so re-running resumes where it left off.

---

## 2. Section-by-Section Breakdown

### 2.1 README.md — Project Overview

- Frames the goal: AstroNet-style transit detection using two fixed-size views:
  - **Global view**: 2001-point "whole orbit" representation
  - **Local view**: 201-point "zoomed-in transit" representation
- Explicitly states that Kaggle datasets are **NOT** used — primary sources only.
- Explains why the pipeline **cannot run inside a sandboxed AI environment**: the NASA Exoplanet Archive and MAST hosts are blocked inside most cloud sandboxes.
- Documents that all mathematical cleaning logic was verified with **synthetic light curves** (injected transits, gaps, NaNs, outliers, stellar-variability wobble).
- Provides a **file-to-stage mapping table** (reproduced and expanded in §5 below).
- Documents sizing expectations: full run ≈ 10–12 GB of raw FITS files.
- Notes all tunable knobs live in `config.py` — no need to touch pipeline logic to change behavior.

---

### 2.2 `config.py` — Configuration Hub

All "magic numbers" are centralized here. Organized into five sections mirroring the four stages plus paths/download:

| Section | Key Constants |
|---|---|
| **Paths** | `PROJECT_ROOT`, `CACHE_DIR`, `DATA_DIR`, `CATALOG_CSV`, `DATASET_NPZ`, `SPLIT_CSV`, `CLEANED_LC_DIR` |
| **Stage A** | `QUALITY_BITMASK = "default"`, `OUTLIER_SIGMA = 5.0` |
| **Stage B** | `FLATTEN_WINDOW_LENGTH = 401`, `FLATTEN_POLYORDER = 2`, `TRANSIT_MASK_PAD_DURATIONS = 1.5` |
| **Stage C** | `GLOBAL_VIEW_BINS = 2001`, `LOCAL_VIEW_BINS = 201`, `LOCAL_VIEW_NUM_DURATIONS = 4.0` |
| **Stage D** | `LABEL_MAP = {"CONFIRMED": 1, "FALSE POSITIVE": 0}`, `MIN_TRANSITS_REQUIRED = 3`, `PERIOD_DEDUPE_TOLERANCE = 0.01` |
| **Download** | `MAX_DOWNLOAD_BYTES = 12 GB`, `MISSION = "Kepler"`, `FLUX_COLUMN = "pdcsap_flux"`, `MAX_TARGETS = None` |
| **Misc** | `RANDOM_SEED = 42` |

---

### 2.3 `catalog.py` — Stage D (Label/Dataset Cleaning)

Handles **everything that happens at the catalog table level** before any individual star's light curve is examined.

**Key functions:**

| Function | Purpose |
|---|---|
| `fetch_koi_catalog()` | Downloads the "cumulative" KOI table from NASA Exoplanet Archive via `astroquery` |
| `clean_catalog(df)` | Applies Stage D cleaning rules #12 and #13 |
| `_harmonic_bucket(periods, tol)` | Helper: detects near-duplicate periods for deduplication |
| `make_class_weights(labels)` | Computes inverse-frequency class weights for training |
| `star_level_split(df)` | Splits train/val/test at the star level (preventing leakage) |
| `load_or_fetch_catalog()` | Convenience wrapper: loads from cache if present, otherwise fetches and cleans |

**Columns extracted from the KOI table:**
`kepid`, `kepoi_name`, `koi_disposition`, `koi_period`, `koi_time0bk`, `koi_duration`, `koi_depth`

---

### 2.4 `lightcurves.py` — Stages A & B (Per-Star Cleaning)

Handles **everything per-star** before any candidate-specific processing:

| Function | Purpose |
|---|---|
| `_dir_size_bytes(path)` | Measures current size of the FITS cache directory |
| `download_budget_remaining()` | Returns bytes remaining before `MAX_DOWNLOAD_BYTES` is hit |
| `cleaned_lightcurve_path(kepid)` | Returns the cache path for a star's cleaned `.fits` file |
| `get_clean_lightcurve(kepid)` | Downloads (or loads cached), stitches, and Stage-A cleans one star's full Kepler mission data |

**Stage A cleaning steps inside `get_clean_lightcurve`:**
1. Quality bitmask applied at download time (removes flagged cadences)
2. PDCSAP_FLUX selected (not raw SAP_FLUX)
3. All quarters stitched together with normalization per quarter
4. NaN flux values removed (`lc.remove_nans()`)
5. Zero/negative flux removed (physically impossible readings)
6. Extreme outliers removed (`lc.remove_outliers(sigma=5.0)`)

---

### 2.5 `views.py` — Stage B#6, Stage C, Stage D#14

Handles **per-(star, candidate) processing** — converts one cleaned light curve + one candidate's ephemeris into fixed-size CNN-ready arrays:

| Function | Purpose |
|---|---|
| `transit_mask(time, period, t0, duration_hours)` | Creates a boolean mask protecting transit cadences from the flattener |
| `count_observed_transits(lc, period, t0)` | Exact re-check of Stage D #14 (minimum transits) using real observed time span |
| `flatten_for_transit(lc, period, t0, duration_hours)` | Stage B #6: removes stellar variability while protecting the transit |
| `_median_bin(x, y, num_bins, x_min, x_max)` | Core binning: median per bin, with linear interpolation for empty bins |
| `_normalize_view(view)` | Stage C #11: median-normalizes flux so absolute star brightness doesn't matter |
| `make_global_view(lc, period, t0)` | Produces the 2001-point whole-orbit view |
| `make_local_view(lc, period, t0, duration_hours)` | Produces the 201-point transit-zoomed view |

---

### 2.6 `dataset.py` — Top-Level Orchestration

Ties everything together:

| Function | Purpose |
|---|---|
| `build_examples(df)` | For each star and each of its candidates: download, clean, build views, collect examples |
| `save_examples(examples)` | Saves the `.npz` dataset, split CSV, and class weights JSON |
| `main()` | CLI entry point with `--max-targets` argument |

---

### 2.7 `run_pipeline.py` — Single Entry Point

A minimal 22-line wrapper. Simply calls `dataset.main()`, which in turn calls `catalog.load_or_fetch_catalog()` first. Exists so users don't have to remember the execution order.

---

### 2.8 `data/` Directory Contents

| File | Description |
|---|---|
| `koi_catalog_clean.csv` | Cleaned, labeled, split KOI catalog (~488 KB, several thousand rows) |
| `star_level_split.csv` | Star-to-split mapping (~4 KB) |
| `transit_dataset.npz.zip` | Final model-ready arrays (compressed, ~2.5 MB) |
| `class_weights.json` | e.g. `{"0": 0.73, "1": 2.14}` — inverse-frequency weights |
| `.gitkeep` | Keeps the empty `data/` directory in version control |

---

## 3. Key Concepts and Definitions

| Term | Definition |
|---|---|
| **Kepler** | NASA space telescope that monitored ~150,000 stars for 4 years, measuring brightness over time |
| **Light curve** | A time series of a star's flux (brightness) measurements over the mission duration |
| **Transit** | The periodic dimming of a star's light caused by a planet passing in front of it |
| **KOI** | Kepler Object of Interest — a candidate transit signal flagged by Kepler's pipeline |
| **TCE** | Threshold Crossing Event — a statistically significant periodic signal in a Kepler light curve |
| **MAST** | Mikulski Archive for Space Telescopes — hosts raw Kepler FITS files |
| **NASA Exoplanet Archive** | Hosts the KOI catalog with human-vetted dispositions |
| **KIC / kepid** | Kepler Input Catalog ID — unique identifier for each observed star |
| **kepoi_name** | Per-candidate ID (e.g. `K00752.01`); one star can have multiple KOIs |
| **PDCSAP_FLUX** | Pre-search Data Conditioned SAP flux — NASA's pipeline already removes instrumental systematics; preferred over raw SAP_FLUX |
| **SAP_FLUX** | Simple Aperture Photometry flux — raw pixel-summed brightness, contains instrument drift |
| **Phase folding** | Aligning all observations by their phase in the orbital period to stack transit signals |
| **Global view** | 2001-bin median-binned representation of the entire orbital period |
| **Local view** | 201-bin zoomed representation of ±(LOCAL_VIEW_NUM_DURATIONS × duration/2) around the transit |
| **AstroNet** | A published CNN architecture for transit detection that inspired this pipeline's two-view approach |
| **Flattening** | Fitting and removing a smooth trend (Savitzky-Golay polynomial) from the light curve to eliminate stellar variability |
| **Class imbalance** | The dataset has far more FALSE POSITIVE examples than CONFIRMED ones — addressed with inverse-frequency class weights |
| **Star-level split** | Ensuring all candidates from the same star are in the same split (train/val/test) to prevent data leakage |
| **GroupShuffleSplit** | scikit-learn class that shuffles data while respecting group integrity (here: star identity) |
| **Harmonic duplicates** | Two TCEs at periods P and 2P (or P/2) that correspond to the same physical signal |
| **BKJD** | Barycentric Kepler Julian Date — the time system used for Kepler timestamps |
| **Ephemeris** | The three parameters that describe a periodic transit: `period`, `t0` (epoch), `duration` |

---

## 4. Important Data and Findings

| Metric | Value | Significance |
|---|---|---|
| Global view size | **2001 bins** | Fixed-length representation of entire orbit; AstroNet standard |
| Local view size | **201 bins** | Fixed-length zoomed transit view; AstroNet standard |
| Local view window | **±4.0 transit durations** | Enough context to capture transit shape without noise |
| Outlier threshold | **5σ** | Removes extreme flux anomalies (cosmic rays, glitches) |
| Flatten window | **401 cadences** | Odd number required by Savitzky-Golay; spans ~20 days at 30-min cadence |
| Flatten poly order | **2** | Quadratic polynomial fitted per window |
| Transit mask padding | **1.5 transit durations** | Protects transit ±1.5 durations either side from the flattener |
| Min transits required | **≥3** | Ensures enough transit events exist to meaningfully phase-fold |
| Period dedupe tolerance | **1%** | Fractional tolerance for collapsing near-identical TCE periods |
| Download cap | **12 GB** | Hard stop on raw FITS file accumulation |
| Typical star file size | **1–3 MB** | One full-mission long-cadence PDCSAP_FLUX light curve |
| Full run size | **~10–12 GB** | Covers all cleaned/labeled KOI host stars |
| Small test run | **~1–2 GB** | Using `--max-targets 300` |
| Random seed | **42** | Ensures reproducible splits |
| LABEL_MAP | `CONFIRMED → 1`, `FALSE POSITIVE → 0` | Binary labels for the CNN |
| CANDIDATE rows | **Dropped** | Ambiguous — no final verdict from NASA |

---

## 5. Processes and Workflows

### 5.1 Complete Pipeline Architecture

```
run_pipeline.py
    └── dataset.main()
            │
            ├── catalog.load_or_fetch_catalog()        ← Stage D (catalog level)
            │       ├── fetch_koi_catalog()            → NASA Exoplanet Archive API
            │       ├── clean_catalog()
            │       │       ├── Drop CANDIDATE rows    (#12)
            │       │       ├── Drop NaN ephem rows
            │       │       └── Dedupe harmonic TCEs   (#13)
            │       ├── make_class_weights()           (#15 — training time only)
            │       └── star_level_split()             (#16)
            │
            └── build_examples(df)                     ← Per-star loop
                    │
                    ├── [Budget check] download_budget_remaining()
                    │
                    ├── lightcurves.get_clean_lightcurve(kepid)
                    │       ├── lk.search_lightcurve() → MAST
                    │       ├── search.download_all(quality_bitmask, pdcsap_flux)  (Stage A #2, B #5)
                    │       ├── collection.stitch()                                (Stage B #7)
                    │       ├── lc.remove_nans()                                   (Stage A #1)
                    │       ├── lc[flux > 0]                                       (Stage A #4)
                    │       └── lc.remove_outliers(sigma=5.0)                      (Stage A #3)
                    │
                    └── [Per-candidate loop]
                            ├── count_observed_transits()   → Stage D #14 (exact check)
                            ├── flatten_for_transit()       → Stage B #6 (variability removal)
                            ├── make_global_view()          → Stage C (2001-bin folded orbit)
                            ├── make_local_view()           → Stage C (201-bin transit zoom)
                            └── Collect → save_examples()
                                    ├── transit_dataset.npz
                                    ├── star_level_split.csv
                                    └── class_weights.json
```

### 5.2 The Four Cleaning Stages

#### Stage A — Bad Measurements (Row-Level Removal)
1. **#1** Remove NaN flux values → `lc.remove_nans()`
2. **#2** Remove quality-flagged cadences → applied at download via `quality_bitmask`
3. **#3** Remove extreme outliers → `lc.remove_outliers(sigma=5.0)`
4. **#4** Remove zero/negative flux → physically impossible readings

#### Stage B — Unwanted Signals (Detrending, Keep All Rows)
5. **#5** Instrument long-term drift → use `PDCSAP_FLUX` not `SAP_FLUX`
6. **#6** Stellar variability → `lc.flatten(window=401, polyorder=2, mask=transit_mask)`
7. **#7** Quarter-to-quarter offsets → `collection.stitch()` normalizes during join

#### Stage C — Structural Problems (Fixed-Size Views)
8. **#8** Data gaps → linear interpolation, never zero-fill
9. **#9** Empty bins after folding → same: linear interpolation
10. **#10** Inconsistent array length → fixed 2001 / 201 bins
11. **#11** Unnormalized flux scale → median-normalize each view: `(flux / median) - 1.0`

#### Stage D — Label/Dataset Cleaning
12. **#12** Ambiguous `CANDIDATE` rows → dropped (no definitive label)
13. **#13** Duplicate/harmonic TCEs → deduped by `(kepid, period_bucket)` keeping first
14. **#14** Too few transits → skip any candidate with fewer than 3 observed transits (re-checked exactly after download)
15. **#15** Class imbalance → NOT deleted; handled by inverse-frequency class weights at training time
16. **#16** Star-level leakage → split by `kepid` so no star spans two splits

### 5.3 Phase Folding Mechanics

The transit signal is recovered by **phase folding**:
1. Compute phase for each time point: `phase = ((time - t0 + 0.5·period) % period) - 0.5·period`
2. Phase now runs from `-period/2` to `+period/2`, with transit at 0
3. Median-bin the folded flux into fixed bins
4. Interpolate any empty bins
5. Normalize by median: `(flux / median) - 1.0`

### 5.4 Transit Masking Logic

```python
half_width = TRANSIT_MASK_PAD_DURATIONS × duration_days / 2.0
phase = ((time - t0 + 0.5·period) % period) - 0.5·period
mask = |phase| < half_width        # True = protect from flattener
```

Padding factor 1.5 means the mask covers ±75% of one transit duration on each side.

### 5.5 Star-Level Split Logic

Uses scikit-learn's `GroupShuffleSplit` twice:
1. First split: 85% `train_val` | 15% `test` (by unique star)
2. Second split on `train_val`: recalculate relative val fraction → `val_size / (1 - test_size)` ≈ 17.6% → yields 85%×0.824 = 70% train, 85%×0.176 = 15% val
3. Final approximate ratio: **70% train / 15% val / 15% test** at the star level
4. Hard assertion confirms no star ever appears in two splits

### 5.6 Class Weight Formula

$$w_c = \frac{N}{n\_classes \times count_c}$$

Where $N$ = total samples, $count_c$ = samples in class $c$. This is standard inverse-frequency weighting compatible with both Keras and PyTorch `class_weight` arguments.

### 5.7 Harmonic Deduplication Algorithm

```
1. Sort candidates by period
2. For each consecutive pair: if |P_i - P_{i-1}| / P_{i-1} < 0.01, assign same bucket
3. drop_duplicates(subset=['kepid', '_period_bucket'], keep='first')
```

> **Note:** This only deduplicates **exact-ish period matches** (within 1%), NOT true harmonics (2P, P/2). The comment in the code acknowledges this is a conservative simplification.

---

## 6. Technical Analysis

### 6.1 Median Binning vs. Zero-Filling

A critical design choice in `_median_bin()`:
- **Empty bins get linearly interpolated** from neighboring non-empty bins.
- Zero-filling would create a **spurious "brightness collapse"** artifact that the CNN could mislearn as a transit signal.
- `np.digitize` assigns each data point to a bin; `np.median` per bin gives robustness to outliers; `np.interp` handles gaps.

### 6.2 Why PDCSAP_FLUX, Not SAP_FLUX

| SAP_FLUX | PDCSAP_FLUX |
|---|---|
| Raw pixel-aperture sum | NASA pipeline pre-processed |
| Contains long-term instrument drift | Corrected for pointing drift, thermal drift |
| Contains crowding contamination | Partially decontaminated |
| Requires detrending by user | Still needs stellar-variability flatten (Stage B #6) |

### 6.3 Download Budget Enforcement

The `download_budget_remaining()` check runs **before each new star's download**:
- Stars already cached in `CLEANED_LC_DIR` are **always processed** (budget has no effect on them).
- Only **new downloads** are halted once `sum(FITS cache files) ≥ 12 GB`.
- This means the effective number of stars processed may slightly exceed the budget if the last batch of downloads pushes it over.

### 6.4 Caching Strategy (Three Layers)
| Layer | Location | What's Cached |
|---|---|---|
| Raw FITS | `lightkurve_cache/` | Raw downloaded quarterly FITS files from MAST |
| Cleaned LC | `data/cleaned_lightcurves/kic{kepid}.fits` | Stage-A-cleaned, stitched light curve per star |
| Catalog | `data/koi_catalog_clean.csv` | Cleaned, split KOI catalog |

### 6.5 Dependencies and Their Roles

| Package | Version | Role |
|---|---|---|
| `lightkurve` | ≥2.4 | Downloads Kepler FITS, provides `LightCurve` objects with `remove_nans`, `remove_outliers`, `stitch`, `flatten`, `fold` |
| `astroquery` | ≥0.4.6 | Queries NASA Exoplanet Archive |
| `astropy` | ≥5.3 | Unit-aware time/flux arrays underlying lightkurve |
| `numpy` | ≥1.24 | Core numerical operations, `.npz` storage |
| `pandas` | ≥2.0 | Catalog DataFrame operations |
| `scipy` | ≥1.10 | Likely used internally by lightkurve for Savitzky-Golay filtering |
| `scikit-learn` | ≥1.3 | `GroupShuffleSplit` for leakage-free splits |

### 6.6 Output Array Specification

The final `transit_dataset.npz` contains:

| Array | Shape | Dtype | Description |
|---|---|---|---|
| `global_view` | (N, 2001) | float32 | Whole-orbit median-binned, normalized flux |
| `local_view` | (N, 201) | float32 | Transit-zoomed median-binned, normalized flux |
| `label` | (N,) | int64 | 0 = FALSE POSITIVE, 1 = CONFIRMED |
| `kepid` | (N,) | int64 | Kepler star ID |
| `split` | (N,) | str/object | "train", "val", or "test" |

---

## 7. Insights and Implications

### Strengths
- **End-to-end reproducibility**: Fixed seed, caching at every layer, idempotent re-runs.
- **No Kaggle dependency**: Direct primary-source downloads ensure data provenance is traceable.
- **Scientifically principled choices**: Using PDCSAP_FLUX, transit-aware masking before flattening, star-level splits — all align with published best practices (AstroNet paper).
- **Separation of concerns**: Each module handles exactly one stage, making the code modular and testable.
- **Defensive design**: Budget cap, cache checks, `None` returns (not exceptions) when data is missing, assertion on split integrity.

### Weaknesses / Limitations
- **Cannot run in sandboxed environments**: MAST and NASA Exoplanet Archive are not accessible from inside AI sandboxes or restricted corporate networks.
- **Harmonic deduplication is simplified**: Only detects near-exact period matches (within 1%), not true 2× or 0.5× harmonics.
- **Full run is slow and large**: 10–12 GB FITS downloads + processing is a multi-hour operation on typical hardware.
- **Stage D #14 is approximate at catalog time**: The minimum-transit check is re-done exactly only after downloading the light curve — some wasted downloads may occur.
- **No data augmentation**: The pipeline produces one view per candidate; augmentation (time-shifting, noise injection) is left to the model training stage.

### Assumptions
- NASA's `koi_disposition` values are taken as ground truth — the pipeline does not independently vet dispositions.
- Long-cadence (30-minute) data is assumed; short-cadence Kepler data is not handled separately.
- A single polynomial degree (2) is assumed adequate for flattening stellar variability across all star types.

### Risks
- If the NASA Exoplanet Archive updates the KOI table (adding new CONFIRMED dispositions), a `--force-refresh` run would produce a different catalog without old results being invalidated.
- The download cap of 12 GB is a hard stop; stars beyond the budget are silently skipped.

---

## 8. Actionable Takeaways

1. **Always run `--max-targets 50` first** to verify the pipeline runs end-to-end before committing to the full 10–12 GB run.
2. **All tuning is in `config.py`**: Change `OUTLIER_SIGMA`, `FLATTEN_WINDOW_LENGTH`, `GLOBAL_VIEW_BINS`, `LOCAL_VIEW_BINS`, `MIN_TRANSITS_REQUIRED`, `QUALITY_BITMASK`, etc. — no other file needs editing.
3. **Re-runs are safe**: Three-layer cache means stopping mid-run does not waste previous work.
4. **Use `class_weights.json` in your loss function**: Do not delete false positives to balance the dataset — use the provided weights with Keras `class_weight=` or PyTorch's weighted sampler.
5. **Never load by split name from the NPZ alone**: Use the `split` array inside the NPZ or `star_level_split.csv` to correctly separate training data.
6. **Do not zero-fill your own extensions**: If you add more views, follow the same `np.interp` pattern for empty bins — zero-filling creates fake CNN features.
7. **Check `data/star_level_split.csv`** to verify no star identity leaked across splits before training.
8. **To use the `CANDIDATE` rows in future**: Add `"CANDIDATE"` to `LABEL_MAP` in `config.py` and assign it a label — the rest of the pipeline will pick it up automatically.
9. **To increase or decrease the download cap**: Change `MAX_DOWNLOAD_BYTES` in `config.py` only.
10. **To switch to short-cadence data**: Change `cadence="long"` to `cadence="short"` in `lightcurves.get_clean_lightcurve()` and adjust `FLATTEN_WINDOW_LENGTH` accordingly (it's in cadences, not time).

---

## 9. Comprehensive Bullet Summary

### Catalog Stage (`catalog.py`)
- Downloads the NASA Exoplanet Archive "cumulative" KOI table via `astroquery`
- Extracts 7 columns: `kepid`, `kepoi_name`, `koi_disposition`, `koi_period`, `koi_time0bk`, `koi_duration`, `koi_depth`
- Drops all `CANDIDATE` rows (ambiguous, no final verdict)
- Drops rows missing `kepid`, `koi_period`, `koi_time0bk`, or `koi_duration`
- Drops rows with `period ≤ 0` or `duration ≤ 0`
- Deduplicates near-identical period TCEs for the same star (within 1% tolerance)
- Maps `CONFIRMED → 1`, `FALSE POSITIVE → 0`
- Computes inverse-frequency class weights (for training, not filtering)
- Performs star-level GroupShuffleSplit into 70/15/15 train/val/test
- Asserts no star spans two splits
- Caches result to `data/koi_catalog_clean.csv`

### Light Curve Stage (`lightcurves.py`)
- Searches MAST for all Kepler long-cadence quarters for each star
- Downloads `PDCSAP_FLUX` (not SAP_FLUX) with quality bitmask filtering
- Stitches all quarters with per-quarter normalization
- Removes NaN, zero/negative, and 5σ outlier flux values
- Caches cleaned light curve to `data/cleaned_lightcurves/kic{kepid}.fits`
- Enforces 12 GB download cap by checking cache size before each new star

### Views Stage (`views.py`)
- For each (star, candidate) pair:
  - Creates a transit mask (±1.5 transit durations) to protect the transit during flattening
  - Flattens stellar variability using Savitzky-Golay polynomial (window=401, order=2) with transit masked out
  - Counts exactly how many transits fall inside the observed time span
  - Skips candidates with fewer than 3 observed transits
  - Phase-folds the light curve using the candidate's period and epoch
  - Median-bins the folded data into 2001 bins (global) and 201 bins (local)
  - Linearly interpolates empty bins (never zero-fills)
  - Median-normalizes each view: `(flux / median) - 1.0`

### Dataset Stage (`dataset.py`)
- Iterates over all unique stars in the catalog
- Checks download budget before each new star
- Collects all valid (global_view, local_view, label, kepid, split) examples
- Saves as compressed `.npz` with float32 views and int64 labels
- Writes per-split counts and class weights
- Logs progress every 50 stars

---

## 10. What Someone Should Remember

> **The most critical facts about this pipeline:**

| # | Key Point |
|---|---|
| 1 | **Two views per candidate**: 2001-bin global (whole orbit) + 201-bin local (transit zoom) — this is the AstroNet architecture input format |
| 2 | **Four cleaning stages A→D**: A=bad rows, B=trends, C=structure, D=labels/splits |
| 3 | **PDCSAP_FLUX is mandatory** — SAP_FLUX contains instrument drift that would corrupt the signal |
| 4 | **Interpolate empty bins, never zero-fill** — zeros look like fake transit signals to a CNN |
| 5 | **Mask transits before flattening** — otherwise the flattener smooths away the very dip being detected |
| 6 | **Star-level splits prevent leakage** — all candidates from one star must be in the same split |
| 7 | **Class imbalance → class weights, not deletion** — `class_weights.json` must be used in the loss function |
| 8 | **CANDIDATE rows are dropped** — only CONFIRMED (1) and FALSE POSITIVE (0) are kept |
| 9 | **All knobs are in `config.py`** — changing behavior never requires editing pipeline logic |
| 10 | **Pipeline is idempotent** — safe to stop and restart; three caching layers ensure no re-work |
| 11 | **Minimum 3 transits required** — checked approximately at catalog stage, re-checked exactly after download |
| 12 | **12 GB download cap** — `MAX_DOWNLOAD_BYTES` in `config.py`; already-cached stars are always processed |
