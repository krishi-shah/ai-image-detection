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
- Evaluate the trained detector across image sets from StyleGAN, Stable Diffusion, Midjourney v6, GPT-4o, and GPT Image 2
- Perform failure case analysis — identify which generative model families fool the detector most and why
- Produce a final written report with findings

---

## Repository Structure

```
ai-image-detection/
├── data/
│   ├── raw/                        # Original downloaded datasets (gitignored)
│   ├── processed/                  # Preprocessed and split datasets (gitignored)
│   └── generalisation/             # Cross-generator evaluation images (gitignored)
│       ├── manifest.json           # Download metadata (source, counts, seed)
│       ├── stylegan/FAKE/          # StyleGAN/StyleGAN2 generated images
│       ├── sd3_flux/FAKE/          # SD3/Flux modern diffusion images
│       ├── midjourney_v6/FAKE/     # Midjourney v6 images
│       ├── gpt4o/FAKE/             # GPT-4o generated images
│       ├── gpt_image_2/FAKE/       # GPT Image 2.0 images (OpenFake OOD test)
│       └── gpt4o_manual/FAKE/      # Manually collected GPT-4o images
├── notebooks/
│   ├── 01_setup_and_eda.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_calibration.ipynb
│   ├── 04_gradcam.ipynb
│   └── 05_generalisation_eval.ipynb  # Milestone 4: cross-generator evaluation
├── scripts/
│   └── download_generalisation_data.py  # Streaming HuggingFace download helper
├── src/
│   ├── model/
│   │   └── detector.py             # EfficientNet-B3 build, checkpoint save/load
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, AUC, ECE, ROC, reliability diagrams
│   │   ├── calibration.py          # Temperature scaling (learned via L-BFGS)
│   │   └── generalisation.py       # Cross-generator evaluation + degradation + plots
│   ├── explainability/
│   │   └── gradcam.py              # Grad-CAM heatmaps + batch failure analysis
│   └── utils/
│       └── data_loader.py          # CIFAKE + generalisation data loaders
├── outputs/
│   ├── checkpoints/                # Saved model weights (gitignored)
│   ├── heatmaps/                   # Grad-CAM output images + failure analysis
│   ├── plots/                      # All evaluation visualisations
│   └── results/                    # baseline_results.json, generalisation_results.json
├── reports/
│   ├── literature_review.md
│   ├── progress_report.md
│   └── progress_report.tex
├── tests/
│   ├── test_metrics.py
│   ├── test_data_loader.py
│   ├── test_detector.py
│   ├── test_calibration.py
│   ├── test_gradcam.py
│   └── test_generalisation.py      # 20 tests for Milestone 4 evaluation module
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
| CIFAKE | Primary training (60K real + AI-generated images) | [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| ForenSynths / CNNDetection | StyleGAN/StyleGAN2 generalisation eval | [OwensLab/CommunityForensics-Eval](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval) (architecture=GAN) |
| Defactify Image Dataset | SD3 modern diffusion eval | [Rajarshi-Roy-research/Defactify_Image_Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) (Label_B=SD3) |
| Defactify / Midjourney Images | Midjourney v6 eval | [Defactify](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) or [ehristoforu/midjourney-images](https://huggingface.co/datasets/ehristoforu/midjourney-images) |
| GPT-ImgEval | GPT-4o autoregressive eval | [Yejy53/GPT-ImgEval](https://huggingface.co/datasets/Yejy53/GPT-ImgEval) |
| OpenFake | GPT Image 2.0 autoregressive successor eval | [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (`core/test`, `model=gpt-image-2.0`) |

---

## Milestone 4: Cross-Generator Generalisation Evaluation

Evaluate the CIFAKE-trained detector on five generator families (StyleGAN, SD3/Flux, Midjourney v6, GPT-4o, GPT Image 2), measure performance degradation, and analyse failure cases with Grad-CAM.

### Prerequisites

- Trained checkpoint at `outputs/checkpoints/best_detector.pth`
- Baseline results at `outputs/results/baseline_results.json`
- CIFAKE real test images at `data/raw/cifake/test/REAL/` (used as the fixed REAL reference for all families)
- HuggingFace login (`huggingface-cli login`) and acceptance of [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) dataset terms (required for GPT Image 2 download)

### Step 1: Download evaluation images

From the project root, download ~300 images per family via streaming (idempotent — skips families that already have enough images). GPT Image 2 may have 200–600 images available in OpenFake; the script downloads `min(300, available)` and the eval loader subsamples REAL images to match.

```bash
python scripts/download_generalisation_data.py --output-dir data/generalisation --per-generator 300
```

On Colab with Google Drive persistence:

```bash
python scripts/download_generalisation_data.py \
  --output-dir data/generalisation \
  --per-generator 300 \
  --base-path /content/drive/MyDrive/ai-image-detection
```

This creates `data/generalisation/manifest.json` and populates:

| Folder | Source |
|---|---|
| `stylegan/FAKE/` | OwensLab/CommunityForensics-Eval (architecture=GAN) |
| `sd3_flux/FAKE/` | Rajarshi-Roy-research/Defactify_Image_Dataset (Label_B=SD3) |
| `midjourney_v6/FAKE/` | Defactify (Label_B=Midjourney) or ehristoforu/midjourney-images |
| `gpt4o/FAKE/` | Yejy53/GPT-ImgEval |
| `gpt_image_2/FAKE/` | ComplexDataLab/OpenFake (`core/test`, `model=gpt-image-2.0`) |
| `gpt4o_manual/` | README for manually collected GPT-4o images |

If a HuggingFace source is gated or unavailable, the script prints manual download instructions and continues with the remaining families.

### Step 2: Run the evaluation notebook

Open `notebooks/05_generalisation_eval.ipynb` on Colab (T4 GPU recommended) or locally. The notebook:

1. Loads the trained checkpoint and baseline results
2. Downloads/verifies generalisation data
3. Evaluates each generator family (accuracy, AUC, ECE, fake detection rate)
4. Computes degradation vs. CIFAKE baseline (96.96% accuracy)
5. Generates Grad-CAM failure heatmaps
6. Saves all plots and JSON results

### Step 3: Run tests

```bash
pytest tests/test_generalisation.py -v
```

Or run the full test suite:

```bash
pytest tests/ -v
```

### Expected outputs

| Path | Description |
|---|---|
| `outputs/results/generalisation_results.json` | Per-family metrics (accuracy, AUC, ECE, F1, confusion matrix) |
| `outputs/results/degradation_summary.json` | Absolute/relative drop vs. CIFAKE baseline |
| `outputs/plots/cross_generator_accuracy.png` | Accuracy bar chart with baseline reference line |
| `outputs/plots/cross_generator_auc.png` | AUC bar chart |
| `outputs/plots/degradation_waterfall.png` | Performance drop waterfall |
| `outputs/plots/confidence_distributions_by_generator.png` | P(FAKE) histograms per family |
| `outputs/plots/ece_comparison_by_generator.png` | ECE pre/post temperature scaling |
| `outputs/plots/gradcam_comparison_grid.png` | Cross-family failure heatmap grid |
| `outputs/heatmaps/gradcam_{family}_failures/` | Per-family false negative + low-confidence heatmaps |
| `outputs/heatmaps/attention_notes.md` | Template for qualitative Grad-CAM observations |

### Key source files

| File | Role |
|---|---|
| `scripts/download_generalisation_data.py` | Streaming HuggingFace download, manifest, idempotent |
| `src/utils/data_loader.py` | `get_generalisation_loader()`, `discover_generator_families()` |
| `src/evaluation/generalisation.py` | `evaluate_generator()`, `compute_degradation()`, plotting |
| `src/explainability/gradcam.py` | `batch_gradcam_failures()`, `comparative_grid()` |

Label convention: **0 = FAKE, 1 = REAL** (matches CIFAKE ImageFolder ordering).

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
