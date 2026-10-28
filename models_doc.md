# EXOSCOPE — The Three Models: What They Are, How They Differ, and What Each Teaches Us

This document covers every model in the project: the **Dual-View CNN** (already built), the
**Baseline Model** (Phase 9, code provided here), and the **Calibrator** (Phase 11, code provided
here). For each: what it is, how it's trained, how it differs from the others, and what a result
from it actually tells you.

---

## 1. Quick Comparison Table

| | **Dual-View CNN** | **Baseline Model** | **Calibrator** |
|---|---|---|---|
| **File** | `model.py` + `train.py` | `baseline_model.py` (+ `features.py`) | `calibrate.py` |
| **Phase** | 10 | 9 | 11 |
| **What it is** | Deep learning (1D CNN) | Classical ML (XGBoost / Logistic Regression) | A 2-parameter logistic regression |
| **Input** | Raw light-curve shape (2001 + 201 numbers) | 6 hand-crafted numbers (depth, duration, period, SNR, odd-even diff, secondary eclipse) | The CNN's own raw output score |
| **Learns** | What a transit *looks like* | Whether simple summary statistics predict the label | Whether the CNN's confidence numbers can be trusted as-is |
| **Trained on** | Train split | Train split | **Validation split only** |
| **Training time** | Minutes (even with 1000s of examples) | Seconds | Seconds |
| **Is this the "final" model?** | Yes — this is the actual product | No — a sanity-check reference point | No — a correction layered on top of the CNN |
| **Can it run alone, in production?** | Yes | Yes (weaker) | No — only makes sense applied *after* the CNN |

---

## 2. What Each Model Actually Is

### 2.1 The Dual-View CNN — the main model
This is the model the entire project exists to build. It looks at the *raw shape* of the light
curve — the actual dip pattern — across two views (a wide "global" view and a zoomed-in "local"
view), and learns on its own what shape features distinguish a real planet from a false positive.
It never sees "depth," "duration," or any other named number directly; it only sees the sequence
of brightness values and has to discover what matters by itself, the same way an image-recognition
CNN learns what an edge or a texture looks like without ever being told the word "edge."

**What a result from this model tells you:** whether a deep learning approach, given enough
examples, can learn the *shape* of a transit well enough to tell real planets from false
positives — the central scientific question of the project.

### 2.2 The Baseline Model — the sanity-check floor
This model never looks at the light curve's shape at all. It only sees **6 single numbers** per
candidate (computed by `features.py`): transit depth, duration, period, signal-to-noise ratio,
odd-even depth difference, and secondary eclipse depth. It uses a classical, well-understood
algorithm (XGBoost, or logistic regression as a fallback) — not deep learning.

**What a result from this model tells you:** a floor for "how good can you get with almost no
effort, using only textbook-simple summary statistics?" If the CNN — which is far more complex,
takes longer to train, and needs far more data — can't comfortably beat this simple model, that's
a strong, specific signal that something is wrong with the CNN's training (not enough data, a bug,
poor hyperparameters), **not** evidence that deep learning "doesn't work" for this problem.

### 2.3 The Calibrator — not really a classifier at all
This is different in kind from the other two. It doesn't look at the light curve, and it doesn't
look at the 6 hand-crafted numbers either. Its *only* input is the CNN's own raw output score, and
its *only* job is to answer: "when the CNN says 80% confident, is it actually right about 80% of
the time?" It's a tiny, 2-parameter logistic regression (slope + intercept) — not a model you'd
ever train or use on its own.

**What a result from this model tells you:** whether the CNN's confidence numbers can be reported
to a scientist as a trustworthy probability, or whether they systematically over- or
under-state how certain the model really is.

---

## 3. Why Three Different Models, Not Just One?

Each model answers a **different question** — they are not three attempts at the same thing:

1. **CNN** → *"Can a deep learning model learn to recognize a transit's shape?"*
2. **Baseline** → *"Is the CNN's complexity actually earning its keep, or would a simple model do
   just as well?"*
3. **Calibrator** → *"Can we trust the CNN's confidence numbers, or do we need to correct them?"*

A project that only built the CNN would have no way to answer questions 2 or 3 — it would have a
number, but no way to know if that number is good, or trustworthy. This is standard, expected
practice in any serious ML project, not extra/optional work.

---

## 4. Dataset → Finished Model: The Process for Each

### 4.1 Dual-View CNN (already built — for reference)

```
transit_dataset.npz
        |
        v
TransitDataset (filtered by split == 'train' / 'val')
        |
        v
DataLoader (batches of 64)
        |
        v
DualViewCNN forward pass -> BCEWithLogitsLoss(pos_weight=...) -> Adam optimizer
        |
        v
30 epochs, saving the best model by validation recall
        |
        v
best_model.pt
```

**Command:** `python train.py`

### 4.2 Baseline Model (Phase 9 — code provided in this response)

```
transit_dataset.npz (via cached cleaned light curves)
        |
        v
features.py: for each candidate, compute depth / duration / period /
             SNR / odd-even diff / secondary eclipse depth
        |
        v
features.csv  (one row per candidate, 6 feature columns + label + split)
        |
        v
baseline_model.py: load features.csv, split by train/val/test,
                    fit XGBoost (or LogisticRegression fallback)
                    with class weighting
        |
        v
baseline_model.joblib + baseline_results.json
         (precision / recall / AUC on train, val, AND test,
          plus feature importances)
```

**Commands, in order:**
```bash
python features.py          # builds features.csv — only needs to run once
python baseline_model.py    # trains + evaluates the baseline model
```

**Important ordering note:** `features.py` must run before `baseline_model.py` — the baseline
model has no light-curve access of its own, it only reads the table `features.py` produces.

### 4.3 Calibrator (Phase 11 — code provided in this response)

```
best_model.pt (the ALREADY-TRAINED CNN — this step does not retrain it)
        |
        v
calibrate.py: run the CNN once over the VALIDATION set only,
              collect its raw sigmoid scores + true labels
        |
        v
Fit a 2-parameter logistic regression: raw_score -> calibrated_probability
        |
        v
calibrator.joblib + calibration_report.json
         (Brier score before/after, reliability curve data)
```

**Command:** `python calibrate.py`

**Important ordering note:** this can only run **after** `train.py` has produced a `best_model.pt`
— there's nothing to calibrate until the CNN itself is trained. It also deliberately never touches
the test set, to keep test results an honest, untouched final check.

---

## 5. What You'll Actually Learn From Each Result

| Result pattern | What it means |
|---|---|
| CNN beats baseline by a wide margin | The deep learning approach is earning its complexity — the raw shape carries more signal than the 6 summary numbers alone |
| CNN barely beats, or loses to, the baseline | A red flag — likely means the CNN needs more data, different hyperparameters, or has a training bug; the simple model is "getting lucky" because the complex one isn't learning enough yet |
| Baseline's feature importances show one feature dominating (e.g. `depth`) | Tells you which single number is most predictive on its own — useful context when tuning Phase 12's physics checks |
| Calibration's Brier score improves after calibration | The CNN's raw confidence was somewhat untrustworthy, and calibration meaningfully fixed it |
| Calibration's Brier score barely changes | The CNN's raw sigmoid output was already fairly well-calibrated — a good sign, though still worth reporting the calibrated number for rigor |

---

## 6. Recommended Run Order (Once Your Big Dataset Finishes)

1. `python train.py` — retrain the CNN on the full dataset
2. `python features.py` — build the hand-crafted feature table (reuses cached light curves, no new downloads)
3. `python baseline_model.py` — train and evaluate the baseline model
4. `python calibrate.py` — calibrate the CNN's output using the validation set

Steps 2–4 are each fast (seconds, not hours) — only step 1 (and the data collection before it)
takes real time.
