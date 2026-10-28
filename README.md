# EXOSCOPE — Automated Exoplanet Transit Vetting Platform

> **Comparative Benchmarking of Hand-Crafted XGBoost, Single-View CNN, and Calibrated Dual-View Multi-Scale Convolutional Architectures on NASA Kepler Light Curves**

[![Vercel Deployment](https://img.shields.io/badge/Deployed%20on-Vercel-black?logo=vercel)](https://vercel.com)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20WebAssembly-blue?logo=webassembly)](https://onnxruntime.ai/)
[![NASA Kepler](https://img.shields.io/badge/Dataset-NASA%20Kepler%20Archive-orange)](https://exoplanetarchive.ipac.caltech.edu/)

---

## 🚀 Overview

**EXOSCOPE** is a production-grade exoplanet vetting platform that ingests raw Kepler photometric time series from MAST, cleans them with astrophysical domain safeguards, and evaluates three comparative modeling paradigms:
1. **Classical Baseline**: XGBoost on 6 hand-crafted astrophysical statistics (76.21% accuracy, 0.805 AUC).
2. **Single-View 1D CNN**: 5 convolutional layers operating on 2,001 phase-folded global time bins (71.30% accuracy, 0.772 AUC).
3. **Flagship Dual-View Multi-Scale CNN**: Two parallel inception branches processing 2,001-point Global View and 201-point Local Transit Zoom simultaneously (77.41% accuracy, 0.826 AUC, Platt Brier score 0.1814).

The web console runs the trained `model.onnx` directly client-side inside the user's browser via **ONNX Runtime Web (WebAssembly)**, with zero server requirements and ~5–15 ms forward pass latency.

---

## 📁 Repository Structure

```text
├── index.html            # Main interactive landing page and live vetting console
├── model.onnx            # Trained DualViewCNN weights exported to ONNX format (186 KB)
├── vercel.json           # Vercel deployment configuration with WebAssembly COOP/COEP headers
├── assets/               # Production diagnostic dashboards & metric figures
│   ├── baseline_metrics_dashboard.png
│   ├── calibration_dashboard.png
│   ├── test_lightcurve_predictions.png
│   ├── test_metrics_dashboard.png
│   └── training_curves.png
└── README.md
```

---

## ⚡ Local Testing

To run locally and allow the browser to load `model.onnx` via WebAssembly:

```bash
# Python 3 local server
python -m http.server 8000
```

Then open `http://localhost:8000` in any modern web browser.

---

## 🌐 Deploy to Vercel (Instant 1-Click)

### Option 1: Via Vercel CLI
```bash
npm install -g vercel
vercel
```

### Option 2: Via GitHub + Vercel Web Dashboard
1. Create a new repository on GitHub (e.g. `exoscope-web`).
2. Push this folder to your repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: EXOSCOPE landing page & in-browser ONNX console"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/exoscope-web.git
   git push -u origin main
   ```
3. Go to [vercel.com/new](https://vercel.com/new) and select the imported repository.
4. Click **Deploy**. Vercel will automatically serve the static site with `vercel.json` headers applied!

---

## 🔬 Scientific Citations

* **Shallue, C. J., & Vanderburg, A. (2018).** *Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90.* The Astronomical Journal, 155(2), 94.
* **Valizadegan, H., et al. (2021).** *ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier That Validates 301 New Exoplanets.* The Astrophysical Journal, 926(2), 120.
* **Platt, J. (1999).** *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.*
