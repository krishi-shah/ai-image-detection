# AI-Generated Image Detection: Evaluating Generalisation Across Generative Model Generations

`[Python 3.10+]` `[PyTorch]` `[EfficientNet-B3]` `[Grad-CAM]` `[Gradio]`

**Course:** EECS 4080 — Computer Science Project, York University  
**Author:** Krishi Rajeshkumar Shah (220968905)  
**Supervisor:** Mona Nasery  
**Term:** Summer 2026

---

## Project Summary

This project investigates the **generalisation gap** in AI-generated image detection: how a detector trained on CIFAKE (Stable Diffusion v1.4) behaves on StyleGAN, SD3/Flux, Midjourney v6, GPT-4o, and Janus-Pro. It combines temperature-scaled calibration, Grad-CAM explainability, frequency/embedding analysis, and an interactive Gradio demo.

## Research Question

> How well do AI-generated image detectors trained on current benchmark datasets generalise to images produced by next-generation generative models, and what accounts for the performance gap?

---

## Key Results

| Metric | CIFAKE Baseline | Cross-generator |
|--------|-----------------|-----------------|
| Test Accuracy | **96.96%** | GPT-4o 92.33%; others ≥96.17% |
| AUC-ROC | **0.9971** | ≥0.9926 on all families |
| Fake detection rate | 95.29% (CIFAKE FAKE recall) | GPT-4o **86.33%**; others 94–97% |
| ECE (pre-calibration) | 0.0026 | Rises under distribution shift |
| Learned Temperature | 1.2189 | Does not transfer OOD |

### Per-Class Performance (CIFAKE Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| FAKE (class 0) | 98.58% | 95.29% | 96.91% | 10,000 |
| REAL (class 1) | 95.44% | 98.63% | 97.01% | 10,000 |

### Key Findings

- In-distribution performance is near-perfect (96.96%, AUC 0.9971).
- Cross-generator transfer is strong (94–97% fake detection) except GPT-4o (86.3%).
- Grad-CAM / FFT / t-SNE support a GPT-4o **feature-absence** story.
- Temperature scaling fitted on CIFAKE **worsens ECE** under distribution shift.

---

## Deliverables

| Deliverable | Path |
|-------------|------|
| Final report | [`reports/final_report.md`](reports/final_report.md) · [`reports/final_report.tex`](reports/final_report.tex) |
| Presentation | [`reports/presentation.pptx`](reports/presentation.pptx) · [`reports/talk_script.md`](reports/talk_script.md) · [`reports/qa_prep.md`](reports/qa_prep.md) |
| Demo runbook | [`reports/demo_script.md`](reports/demo_script.md) (Q&A backup only) |
| Setup guide | [`docs/SETUP.md`](docs/SETUP.md) |
| Deliverables index | [`reports/DELIVERABLES.md`](reports/DELIVERABLES.md) |
| Gradio demo | [`app.py`](app.py) |

---

## Repository Structure

```
ai-image-detection/
├── app.py
├── requirements.txt
├── LICENSE
├── README.md
├── docs/SETUP.md
├── data/
│   ├── raw/                    # CIFAKE (gitignored)
│   ├── demo_samples/
│   └── generalisation/         # Cross-generator eval sets
├── notebooks/
│   ├── 01_setup_and_eda.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_calibration.ipynb
│   ├── 04_gradcam.ipynb
│   ├── 05_generalisation_eval.ipynb
│   ├── 06_gpt4o_investigation.ipynb
│   └── 06_gpt4o_investigation.ipynb
├── scripts/
│   ├── download_generalisation_data.py
│   └── build_presentation.py
├── src/
│   ├── model/detector.py
│   ├── evaluation/{metrics,calibration,generalisation}.py
│   ├── explainability/gradcam.py
│   ├── analysis/{frequency,embeddings,gradcam_metrics}.py
│   └── utils/data_loader.py
├── outputs/
├── reports/
└── tests/                      # 63 tests across 7 modules
```

---

## Setup

See **[`docs/SETUP.md`](docs/SETUP.md)** for full instructions. Quick start:

```bash
git clone https://github.com/krishi-shah/ai-image-detection.git
cd ai-image-detection
pip install -r requirements.txt
```

1. Download CIFAKE into `data/raw/cifake/` (`train/` and `test/` with `REAL/` and `FAKE/`).
2. Place checkpoint at `outputs/checkpoints/best_detector.pth`.
3. Launch demo: `python app.py`

Colab: open any notebook → Runtime → T4 GPU.

---

## Milestones

### Milestone 1 — Literature & Setup (May 1–20)
Literature review, repo, CIFAKE EDA. → [`reports/literature_review.md`](reports/literature_review.md)

### Milestone 2 — Baseline & Calibration (May 21–June 15)
EfficientNet-B3 fine-tune, temperature scaling. → notebooks `02`, `03`

### Milestone 3 — Grad-CAM (June 16–July 5)
Heatmaps on FAKE vs REAL. → notebook `04`

### Milestone 4 — Generalisation (July 6–20)
Five-family evaluation + GPT-4o deep dive. → notebooks `05`, `06` · [`reports/progress_report.tex`](reports/progress_report.tex)

### Milestone 5 — Gradio Demo (July 21–Aug 10)
Interactive classification + confidence + Grad-CAM. → `app.py`

### Milestone 6 — Final Report (Aug 11–28)
Final report, presentation, docs. → `reports/final_report.*` · `reports/presentation.pptx`

---

## Cross-Generator Summary

| Generator | Accuracy | Fake Det. Rate | Acc. Drop (rel) |
|-----------|----------|----------------|-----------------|
| CIFAKE baseline | 0.9696 | — | 0% |
| GPT-4o | **0.9233** | **0.8633** | **4.8%** |
| Janus-Pro | 0.9617 | 0.9400 | 0.8% |
| Midjourney v6 | 0.9617 | 0.9400 | 0.8% |
| SD3/Flux | 0.9767 | 0.9700 | −0.7% |
| StyleGAN | 0.9617 | 0.9400 | 0.8% |

---

## Evaluation Metrics

- **Accuracy** — overall classification accuracy
- **AUC-ROC** — threshold-independent discrimination
- **ECE** — Expected Calibration Error
- **Fake Detection Rate** — recall on FAKE images (primary OOD metric)
- **Reliability diagrams** — visual calibration quality

---

## Running Tests

```bash
# Avoid broken third-party pytest plugins on some machines
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'   # PowerShell
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   # bash

pytest tests/ -v
```

**63 tests** across 7 modules (detector, data loader, metrics, calibration, Grad-CAM, generalisation, app).

---

## Milestone Timeline

| Period | Milestone | Focus |
|--------|-----------|-------|
| May 1 – May 20 | 1 | Literature, repo, CIFAKE |
| May 21 – June 15 | 2 | Baseline + calibration |
| June 16 – July 5 | 3 | Grad-CAM |
| July 6 – July 20 | 4 | Generalisation study |
| July 21 – Aug 10 | 5 | Gradio demo |
| Aug 11 – Aug 28 | 6 | Final report and presentation |

---

## References

- Bird, J.J. & Lotfi, A. (2024). CIFAKE… *IEEE Access*.
- Corvi, R. et al. (2023). Detection of Diffusion Synthetic Images. *ICASSP*.
- Gragnaniello, D. et al. (2021). Are GAN Generated Images Easy to Detect? *ICME*.
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
- Rossler, A. et al. (2019). FaceForensics++. *ICCV*.
- Selvaraju, R. et al. (2017). Grad-CAM. *ICCV*.
- Wang, S. et al. (2020). CNN-Generated Images Are Surprisingly Easy to Spot… *CVPR*.
