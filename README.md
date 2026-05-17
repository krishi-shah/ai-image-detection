# AI-Generated Image Detection: Evaluating Generalisation Across Generative Model Generations

`[Python 3.10]` `[PyTorch]` `[EfficientNet-B3]` `[Grad-CAM]` `[Gradio]`

---

## Project Summary

This project investigates the generalisation gap in AI-generated image detection. Detectors trained on older generative model outputs — such as early GAN architectures — may fail on images produced by newer models because they learned low-level statistical artifacts (e.g., spectral signatures, checkerboard patterns, or upsampling traces) that newer architectures, including latent diffusion models and autoregressive generators like GPT-4o, no longer produce. By fine-tuning an EfficientNet-B3 classifier on the CIFAKE benchmark and then evaluating it against multiple generations of generative models, this project quantifies the performance gap and uses Grad-CAM visualisations and calibration analysis to identify what features the detector actually learned — and why those features fail to transfer.

---

## Research Question

"How well do AI-generated image detectors trained on current benchmark datasets generalise to images produced by next-generation generative models, and what accounts for the performance gap?"

---

## Project Phases

**Phase 1 (May – July 5): Baseline Pipeline**
- Train a binary EfficientNet-B3 classifier on the CIFAKE dataset (60K real + AI-generated images)
- Apply temperature scaling for post-hoc confidence calibration
- Integrate Grad-CAM for visual explainability of model decisions

**Phase 2 (July 6 onward): Generalisation Evaluation**
- Evaluate the trained detector across image sets from StyleGAN, Stable Diffusion, Midjourney v6, and GPT-4o
- Perform failure case analysis — identify which generative model families fool the detector most and why
- Produce a final written report with findings

---

## Repository Structure

```
ai-image-detection/
├── data/
│   ├── raw/                        # Original downloaded datasets (gitignored)
│   └── processed/                  # Preprocessed and split datasets (gitignored)
├── notebooks/
│   ├── 01_setup_and_eda.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_calibration.ipynb
│   ├── 04_gradcam.ipynb
│   └── 05_generalisation_eval.ipynb
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── detector.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── gradcam.py
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py
├── outputs/
│   ├── checkpoints/                # Saved model weights (gitignored)
│   ├── heatmaps/                   # Grad-CAM output images
│   ├── plots/                      # Reliability diagrams, ROC curves
│   └── results/                    # Evaluation CSVs and JSON logs
├── reports/
│   ├── literature_review.md
│   ├── progress_report.md
│   └── final_report.md
├── tests/
│   ├── test_metrics.py
│   └── test_data_loader.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/krishi-shah/ai-image-detection.git
   cd ai-image-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the CIFAKE dataset**
   Download from Kaggle: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
   Place the extracted contents in `data/raw/cifake/` so the directory contains `train/` and `test/` subdirectories.

4. **Google Colab (recommended for GPU access)**
   Open any notebook in Colab. Then: Runtime > Change runtime type > T4 GPU.

---

## Datasets

| Dataset | Purpose | Source |
|---|---|---|
| CIFAKE | Primary training (60K real + AI-generated images) | Kaggle |
| GenImage | Generalisation eval (DALL-E, Midjourney, Stable Diffusion) | HuggingFace |
| FaceForensics++ | Supplementary facial manipulation testing | GitHub |
| GPT-4o / MJ v6 outputs | Newest generation test set | Collected manually |

---

## Evaluation Metrics

- **Accuracy** — overall classification accuracy on balanced test sets
- **AUC-ROC** — area under the receiver operating characteristic curve; threshold-independent performance measure
- **ECE (Expected Calibration Error)** — measures how well the model's confidence scores match true accuracy; lower is better
- **Reliability Diagrams** — visual plots of calibration quality; perfect calibration lies on the diagonal

---

## Milestone Timeline

| Period | Focus |
|---|---|
| May 1 – May 20 | Literature review, repo setup, CIFAKE preprocessing, metric definition |
| May 21 – June 15 | EfficientNet-B3 fine-tuning, temperature scaling, initial eval |
| June 16 – July 5 | Grad-CAM integration, calibration refinement, reproducible pipeline |
| July 6 – July 20 | Generalisation study across model generations, progress report |
| July 21 – Aug 10 | Gradio demo development and integration |
| Aug 11 – Aug 28 | Final report, codebase cleanup, submission |

---

## Key References

- Bird & Lotfi (2024). CIFAKE. IEEE Access, 12, 15642-15650.
- Wang et al. (2020). CNN-Generated Images Are Surprisingly Easy to Spot...For Now. CVPR.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
- Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Corvi et al. (2023). On the Detection of Synthetic Images Generated by Diffusion Models. ICASSP.
- Rossler et al. (2019). FaceForensics++. ICCV.
- Gragnaniello et al. (2021). Are GAN Generated Images Easy to Detect? ICME.
