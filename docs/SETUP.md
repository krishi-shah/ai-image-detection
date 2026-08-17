# Setup Guide

**Project:** AI-Generated Image Detection — Evaluating Generalisation Across Generative Model Generations  
**Repo:** https://github.com/krishi-shah/ai-image-detection

This guide covers local setup, Google Colab, dataset acquisition, checkpoints, notebooks, the Gradio demo, and the test suite.

---

## 1. Clone and install

```bash
git clone https://github.com/krishi-shah/ai-image-detection.git
cd ai-image-detection
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10+ is recommended. The project was developed primarily on Colab (Python 3.10/3.11) and verified locally on Windows.

---

## 2. CIFAKE dataset

1. Download from [Kaggle — CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
2. Extract so the layout is:

```
data/raw/cifake/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

The archive used in this project contains **100,000** training images (50K REAL + 50K FAKE) and **20,000** test images (10K + 10K). The paper/Kaggle card often cites 60,000; all reported experiments use the on-disk counts above.

`data/raw/` is gitignored.

---

## 3. Model checkpoint

Place the trained weights at:

```
outputs/checkpoints/best_detector.pth
```

On Colab, the notebooks also look under Google Drive:

```
/content/drive/MyDrive/ai-image-detection/checkpoints/best_detector.pth
```

Temperature and baseline metrics live in `outputs/results/baseline_results.json` (learned T = 1.2189).

---

## 4. Google Colab (recommended for GPU)

1. Upload the CIFAKE zip to Drive under `MyDrive/ai-image-detection/`.
2. Open a notebook from `notebooks/` in Colab.
3. **Runtime → Change runtime type → T4 GPU**.
4. Run cells top to bottom. The setup cells mount Drive, clone the repo, install requirements, and extract CIFAKE.

---

## 5. Notebooks (run order)

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_eda.ipynb` | Environment + CIFAKE EDA |
| `02_baseline_training.ipynb` | Hyperparameter sweep + full training |
| `03_calibration.ipynb` | Temperature scaling, threshold analysis |
| `04_gradcam.ipynb` | Grad-CAM heatmaps |
| `05_generalisation_eval.ipynb` | Cross-generator evaluation |
| `06_gpt4o_investigation.ipynb` | FFT, t-SNE, Grad-CAM metrics |
| `06_gpt4o_investigation.ipynb` | FFT, t-SNE, Grad-CAM metrics |

### Generalisation data (notebooks 05–06)

```bash
python scripts/download_generalisation_data.py --output-dir data/generalisation --per-generator 300
```

---

## 6. Gradio demo

```bash
python app.py                 # http://localhost:7860
python app.py --share         # public gradio.live URL (Colab-friendly)
```

Requires the checkpoint and `baseline_results.json`. If the checkpoint is missing, the app starts and shows setup instructions instead of crashing.

Demo samples: `data/demo_samples/`.

---

## 7. Tests

```bash
# Recommended (avoids broken third-party pytest plugins)
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   # Windows cmd
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'  # PowerShell
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1  # bash

pytest tests/ -v
```

All tests should pass (63 across 7 modules).

### Troubleshooting: `ModuleNotFoundError: No module named 'mlcheck'`

A stray `mlcheck` pytest plugin on some machines auto-loads and crashes collection. Setting `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` disables plugin auto-discovery and fixes this.

---

## 8. Presentation rebuild

```bash
pip install python-pptx
python scripts/build_presentation.py
```

Writes `reports/presentation.pptx` (13 slides, speaker notes). Spoken lines: `reports/talk_script.md`. Q&A: `reports/qa_prep.md`.

---

## 9. Final report PDF

Open `reports/final_report.tex` in Overleaf (or a local TeX install) with the `outputs/` figure tree available relative to the project root.
