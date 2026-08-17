# AI-Generated Image Detection

**Evaluating generalisation across generative model generations**

An EfficientNet-B3 detector trained on CIFAKE (Stable Diffusion v1.4), with temperature scaling and Grad-CAM explainability, evaluated on StyleGAN, SD3/Flux, Midjourney v6, GPT-4o, and Janus-Pro.

In-distribution performance is strong (**96.96%** accuracy, AUC **0.9971**, ECE **0.0026**). Cross-generator transfer is robust (**94–97%** fake detection) except for **GPT-4o (86.3%)**. Frequency analysis, t-SNE, and Grad-CAM support a GPT-4o **feature-absence** story: the detector attends to reasonable locations but finds no Stable Diffusion v1.4 artefacts. Temperature scaling fitted in-distribution does not remain optimal under shift.

`[Python 3.10+]` `[PyTorch]` `[EfficientNet-B3]` `[Grad-CAM]` `[Gradio]`

**EECS 4080** — Computer Science Project, York University  
**Author:** Krishi Rajeshkumar Shah (220968905) · **Supervisor:** Mona Nasery · **Term:** Summer 2026  
**Repo:** https://github.com/krishi-shah/ai-image-detection

---

## Demo

The Gradio app classifies an image as REAL or FAKE, reports temperature-scaled confidence, and overlays a Grad-CAM heatmap. One-click samples: CIFAKE Real, CIFAKE Fake, StyleGAN, Midjourney, and GPT-4o.

![Gradio demo of AI image detection](assets/demo.mp4)

```bash
git clone https://github.com/krishi-shah/ai-image-detection.git
cd ai-image-detection
pip install -r requirements.txt
```

1. Download [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) into `data/raw/cifake/` (`train/` and `test/` with `REAL/` and `FAKE/`).
2. Place the checkpoint at `outputs/checkpoints/best_detector.pth`.
3. Launch:

```bash
python app.py                 # http://localhost:7860
python app.py --share         # public gradio.live URL
```

Full setup (venv, Colab, datasets, tests): **[`docs/SETUP.md`](docs/SETUP.md)**

---

## Research Question

> How well do AI-generated image detectors trained on current benchmark datasets generalise to images produced by next-generation generative models, and what accounts for the performance gap?

**Hypothesis.** Detectors rely on low-level statistical artefacts specific to the training generator. Those artefacts are weaker or absent in newer models, producing a measurable generalisation gap that widens with generator novelty.

AI-image detectors are mostly trained on older generators. As SD3, Midjourney v6, and GPT-4o close the perceptual gap with photographs, the question is whether existing detectors still work — and if not, why.

**Research gap.** Prior work covers GAN fingerprints (Wang et al., 2020), CIFAKE detection with LIME (Bird & Lotfi, 2024), and a GAN–diffusion generalisation gap (Corvi et al., 2023). This project combines (1) five chronologically distinct generator families, (2) calibration alongside accuracy, (3) Grad-CAM compared across families, and (4) measuring the generalisation problem rather than proposing a new detector.

---

## Datasets

The training set is [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (Bird & Lotfi, 2024): CIFAR-10 photos vs Stable Diffusion v1.4, native **32×32**, upscaled to **224×224**. Labels follow ImageFolder alphabetical order: **FAKE = 0**, **REAL = 1**.

| Quantity | Count | What it is |
|----------|-------|------------|
| On disk | 120,000 | Kaggle archive: 100k `train/` + 20k `test/` |
| By class | 60,000 REAL + 60,000 FAKE | CIFAR-10 vs SD v1.4 |
| Train / val | 80,000 + 20,000 | 80/20 split of `train/` (seed 42) |
| Held-out test | 20,000 | the separate `test/` folder |

Paper / Kaggle cards often say “60,000”; that is one class or an older card, not this archive. 80k + 20k + 20k = 120k. Nothing is missing and nothing is double-counted.

| Split | FAKE | REAL | Total | % FAKE |
|-------|------|------|-------|--------|
| Train | 39,962 | 40,038 | 80,000 | 50.0% |
| Validation | 10,038 | 9,962 | 20,000 | 50.2% |
| Test | 10,000 | 10,000 | 20,000 | 50.0% |

Cross-generator fakes were streamed from HuggingFace (not GenImage). Each family is 300 FAKE images paired with 300 CIFAKE `test/REAL` images (seeded permutation). *N* = 300 is enough to see GPT-4o versus the other families; it is not enough to rank Midjourney against StyleGAN.

| Set | Role | Source |
|-----|------|--------|
| CIFAKE | train / val / test | Kaggle (Bird & Lotfi, 2024) |
| StyleGAN | 300 FAKE | CommunityForensics (GAN) |
| SD3/Flux | 300 FAKE | Defactify Image Dataset |
| Midjourney v6 | 300 FAKE | Defactify Image Dataset |
| GPT-4o | 300 FAKE | GPT-ImgEval (Yejy53) |
| Janus-Pro | 300 FAKE | Janus-Pro-R1-Data (midbee) |
| CIFAKE `test/REAL` | 300 REAL per family | CIFAKE test folder |

Because half of each eval set is in-distribution CIFAKE REAL, overall accuracy is inflated by easy reals. **Fake detection rate (FAKE recall)** is the primary out-of-distribution metric.

---

## Method

### Architecture

ImageNet-pretrained **EfficientNet-B3** (`timm`), classifier replaced with `Linear(1536, 2)`:

```
EfficientNet-B3 (ImageNet pretrained)
 └── classifier: Linear(1536, 2)
```

Binary logits `[FAKE, REAL]` → softmax. Classified as FAKE if *P*(FAKE) > 0.5.

### Calibration

Post-hoc **temperature scaling** (Guo et al., 2017): a scalar *T* is learned on the validation set by minimising NLL with L-BFGS (init 1.5, up to 50 iterations):

```
calibrated probs = softmax(logits / T)
```

Learned **T = 1.2189** (*T* > 1 softens a slightly overconfident model).

### Explainability

**Grad-CAM** (`pytorch-grad-cam`) on EfficientNet-B3’s final convolutional head (`model.conv_head`).

### Training

A 6-run sweep (3 learning rates × 2 weight decays, 3 epochs each) selected the pair with highest validation accuracy for a 20-epoch run.

| Parameter | Value |
|-----------|-------|
| Sweep LRs | 3×10⁻⁴, 10⁻⁴, 5×10⁻⁵ |
| Sweep weight decays | 10⁻⁴, 10⁻⁵ |
| Selected LR | 3×10⁻⁴ (best 3-epoch val acc 0.9594) |
| Selected weight decay | 10⁻⁴ |
| Epochs | 20 (best checkpoint: **epoch 2**, val acc 0.9694) |
| Batch size | 32 |
| Optimiser | AdamW |
| Scheduler | CosineAnnealingLR (*T*<sub>max</sub> = 20) |
| Loss | CrossEntropyLoss |
| Seed | 42 |

CIFAKE is easy for ImageNet-pretrained EfficientNet-B3 (strong SD v1.4 texture at 32×32), so validation peaks early. Later epochs would overfit. All evaluation loads `outputs/checkpoints/best_detector.pth`, not the last epoch.

**Train transforms** (identical for both classes): Resize(224) → RandomHorizontalFlip → RandomCrop(224, padding=8) → ColorJitter (brightness/contrast/saturation 0.2, hue 0.1) → ToTensor → ImageNet normalize.

**Val / test transforms:** Resize(256) → CenterCrop(224) → ToTensor → same ImageNet normalize.

Both classes share one `ImageFolder` transform pipeline. There is no class-conditional preprocessing.

---

## In-Distribution Results (CIFAKE test, 20,000 images)

| Metric | Value |
|--------|-------|
| Test accuracy | **96.96%** |
| AUC-ROC | **0.9971** |
| ECE (before calibration) | **0.0026** |
| ECE (after calibration) | 0.0113 |
| Learned temperature | 1.2189 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| FAKE (0) | 0.9858 | 0.9529 | 0.9691 | 10,000 |
| REAL (1) | 0.9544 | 0.9863 | 0.9701 | 10,000 |
| Accuracy | | | 0.9696 | 20,000 |

Confusion matrix (positive class = FAKE):

| | Predicted FAKE | Predicted REAL |
|--|----------------|----------------|
| Actual FAKE | 9,529 (TP) | 471 (FN) |
| Actual REAL | 137 (FP) | 9,863 (TN) |

Predicted probabilities are well separated: mean *P*(FAKE) is **0.9293** on actual fakes vs **0.0244** on actual reals (medians 0.9983 vs 0.0000).

### Threshold

All metrics stay above 91% from threshold 0.40 to 0.70. Accuracy peaks at 0.40 (**97.22%**), but **0.50 is kept**: the gap is 0.26%, a lower threshold raises false positives (precision 0.9858 → 0.9635), and 0.50 is the natural “more confident than not” boundary.

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 0.40 | 0.9722 | 0.9807 | 0.9635 | 0.9720 |
| 0.45 | 0.9716 | 0.9839 | 0.9589 | 0.9712 |
| **0.50** | **0.9696** | **0.9858** | **0.9529** | **0.9691** |
| 0.55 | 0.9665 | 0.9871 | 0.9454 | 0.9658 |
| 0.60 | 0.9646 | 0.9894 | 0.9393 | 0.9637 |
| 0.65 | 0.9605 | 0.9906 | 0.9297 | 0.9592 |
| 0.70 | 0.9563 | 0.9922 | 0.9197 | 0.9546 |

### Calibration

The model was already well calibrated (test ECE 0.0026). Temperature scaling optimises NLL, not ECE, and **over-corrected**: ECE rose on both validation (0.0031 → 0.0116) and test (0.0026 → 0.0113).

| Set | ECE before | ECE after | *T* |
|-----|------------|-----------|-----|
| Validation (used to learn *T*) | 0.0031 | 0.0116 | 1.2189 |
| Test (held-out) | 0.0026 | 0.0113 | 1.2189 |

---

## Analysis

**Misclassified fakes.** 471 / 10,000 FAKE images were called REAL (4.71% FN). The most confident misses look more like photographs: fewer SD v1.4 artefacts (oversmoothing, colour banding, texture repetition), more natural composition and colour. Grad-CAM shows the model leans on those low-level textures; when a fake happens to lack them, there is little signal left. That is consistent with Gragnaniello et al. (2021) and with the project hypothesis.

**Grad-CAM.** REAL images get broad, diffuse texture attention at 99–100% confidence. FAKE images get localised artefact hotspots; lower-confidence fakes (70–72%) show scattered activation. The detector has learned **distribution-specific SD v1.4 features** that may not transfer.

---

## Cross-Generator Generalisation

Five families, 300 FAKE each, paired with 300 CIFAKE test reals. Every REAL is a 32×32 CIFAR upscale; every FAKE is a high-resolution image downscaled to model input. Use **fake detection rate**, not accuracy, as the OOD metric.

| Generator | Acc. | Fake det. | AUC | F1 (FAKE) | ECE pre-*T* | ECE post-*T* | Rel. drop |
|-----------|------|-----------|-----|-----------|-------------|--------------|-----------|
| CIFAKE (baseline) | 0.9696 | — | 0.9971 | — | 0.0026 | 0.0113 | 0.0% |
| **GPT-4o** | **0.9233** | **0.8633** | **0.9926** | **0.9184** | 0.0083 | 0.0209 | **4.8%** |
| Janus-Pro | 0.9617 | 0.9400 | 0.9958 | 0.9608 | 0.0199 | 0.0334 | 0.8% |
| Midjourney v6 | 0.9617 | 0.9400 | 0.9955 | 0.9608 | 0.0338 | 0.0454 | 0.8% |
| SD3/Flux | 0.9767 | 0.9700 | 0.9975 | 0.9765 | 0.0229 | 0.0363 | −0.7% |
| StyleGAN | 0.9617 | 0.9400 | 0.9945 | 0.9608 | 0.0342 | 0.0514 | 0.8% |

1. **GPT-4o is the only meaningful drop.** 92.33% accuracy, 86.33% fake detection — nearly 14% of GPT-4o images fooled the detector. All other families stayed at or above 96% accuracy.
2. **The autoregressive hypothesis was partly wrong.** GPT-4o and Janus-Pro are both autoregressive, but Janus-Pro matched StyleGAN and Midjourney (96.17%). The gap is GPT-4o-specific.
3. **Diffusion-to-diffusion transfer is strong.** SD3/Flux exceeded the CIFAKE baseline.
4. **Calibration does not transfer.** CIFAKE-fitted *T* = 1.2189 worsened ECE on every OOD set (post-*T* ECE 0.0209–0.0514 vs 0.0113 on CIFAKE). Fit calibration per target distribution, or use methods that survive shift.

---

## GPT-4o Investigation

Three follow-ups at frequency, representation, and attention:

1. **FFT radial power spectrum** — grayscale 2-D FFT, radial average, 150 images per family. Tests whether GPT-4o lacks SD v1.4 frequency artefacts.
2. **t-SNE** — 1,536-d penultimate embeddings (global average pooling hook), 150 images per family, perplexity 30, seed 42. Tests whether GPT-4o clusters with CIFAKE REAL (looks real to the network) or CIFAKE FAKE (threshold/calibration issue).
3. **Quantitative Grad-CAM** — entropy, peak activation ratio (top 10% of pixels), and Gini on 100 FAKE images per family.

| Family | Mean entropy | Mean peak ratio | Mean Gini |
|--------|--------------|-----------------|-----------|
| CIFAKE FAKE | 10.197 | 0.339 | 0.574 |
| GPT-4o | 10.291 | 0.309 | 0.529 |
| Janus-Pro | 10.355 | 0.295 | 0.498 |
| Midjourney v6 | 10.376 | 0.282 | 0.478 |
| SD3/Flux | 10.380 | 0.282 | 0.488 |
| StyleGAN | 10.319 | 0.317 | 0.518 |

CIFAKE FAKE has the most focused attention (lowest entropy, highest Gini). GPT-4o’s attention metrics are **not** dramatically worse than well-detected families — Midjourney and SD3/Flux have higher entropy and lower peak ratios, yet 96–97% accuracy.

That rules out “the model does not know where to look.” The degradation is **feature absence**: it attends to reasonable locations but finds no recognisable SD v1.4 artefacts. GPT-4o’s rendering pipeline does not produce those spectral fingerprints.

---

## Hypothesis Assessment

| Claim | Outcome |
|-------|---------|
| Detectors fail on newer generators in general | Partially supported — only GPT-4o drops; others transfer at 94–97% fake detection |
| Autoregressive generators are especially hard | Partially supported (GPT-4o is hard; Janus-Pro matches StyleGAN / Midjourney) |
| Failure due to missing training artefacts | Supported for GPT-4o (FFT, t-SNE, Grad-CAM) |
| Calibration transfers under shift | **Refuted** |

A second benchmark (e.g. GenImage) and several detector architectures were not run. The question is how a **CIFAKE-trained** detector behaves on newer generators. GenImage does not cover GPT-4o, Janus-Pro, or Midjourney v6; five streamed families were the substitute. One ordinary backbone was used on purpose: if a standard detector fails, the failure is the data and the test, not a weak network.

---

## Limitations

1. **Single-generator 32×32 training.** Results may not transfer to other backbones or to detectors trained on higher-resolution mixed-generator data.
2. **Sample size.** 300 images per family can separate GPT-4o from the rest; it cannot rank Midjourney vs StyleGAN.
3. **CIFAKE-real pairing.** Accuracy is inflated by easy in-distribution reals. Use fake detection rate OOD.
4. **Calibration does not transfer.** Do not treat OOD confidence bars as true probabilities.
5. **Single architecture.** EfficientNet-B3 may rely on cues other backbones do not.

**Future work:** train on multi-generator, higher-resolution data; add hybrid spatial–frequency features; evaluate ensembles; refit calibration per deployment distribution.

---

## Repository

```
ai-image-detection/
├── app.py                      # Gradio demo
├── assets/demo.mp4             # Demo recording
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
│   └── 06_gpt4o_investigation.ipynb
├── src/
│   ├── model/detector.py
│   ├── evaluation/             # metrics, calibration, generalisation
│   ├── explainability/gradcam.py
│   ├── analysis/               # frequency, embeddings, Grad-CAM metrics
│   └── utils/data_loader.py
├── outputs/                    # results JSON, heatmaps (checkpoint gitignored)
└── tests/                      # 63 tests across 7 modules
```

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_eda.ipynb` | Environment, CIFAKE download, EDA |
| `02_baseline_training.ipynb` | Hyperparameter sweep and full training |
| `03_calibration.ipynb` | Temperature scaling and thresholds |
| `04_gradcam.ipynb` | Grad-CAM heatmaps |
| `05_generalisation_eval.ipynb` | Five-family evaluation |
| `06_gpt4o_investigation.ipynb` | FFT, t-SNE, Grad-CAM metrics |

| Module | Role |
|--------|------|
| `src/model/detector.py` | EfficientNet-B3 build, checkpoint I/O |
| `src/evaluation/metrics.py` | Accuracy, AUC, ECE, ROC |
| `src/evaluation/calibration.py` | Temperature scaling (L-BFGS) |
| `src/evaluation/generalisation.py` | Cross-generator eval and plots |
| `src/explainability/gradcam.py` | Heatmaps and comparison grids |
| `src/analysis/frequency.py` | FFT radial power spectra |
| `src/analysis/embeddings.py` | t-SNE of penultimate embeddings |
| `src/analysis/gradcam_metrics.py` | Entropy, peak ratio, Gini |
| `src/utils/data_loader.py` | CIFAKE and family loaders |
| `app.py` | Gradio demo |

Colab: open any notebook → Runtime → T4 GPU.

### Tests

```bash
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'   # PowerShell
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   # bash

pytest tests/ -v
```

63 tests across 7 modules (detector, data loader, metrics, calibration, Grad-CAM, generalisation, app).

### Timeline

| Milestone | Period | Focus |
|-----------|--------|-------|
| 1 | May 1–20 | Literature, repo, CIFAKE EDA |
| 2 | May 21–Jun 15 | EfficientNet-B3 + temperature scaling |
| 3 | Jun 16–Jul 5 | Grad-CAM, ECE, threshold analysis |
| 4 | Jul 6–20 | Five-family eval and GPT-4o investigation |
| 5 | Jul 21–Aug 10 | Gradio demo |
| 6 | Aug 11–28 | Final report and presentation |

---

## References

1. Bird, J.J. & Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. *IEEE Access*, 12, 15642–15650.
2. Corvi, R. et al. (2023). On the Detection of Synthetic Images Generated by Diffusion Models. *ICASSP*.
3. Gragnaniello, D. et al. (2021). Are GAN Generated Images Easy to Detect? A Critical Analysis of the State of the Art. *ICME*.
4. Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
5. Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
6. Wang, S. et al. (2020). CNN-Generated Images Are Surprisingly Easy to Spot… For Now. *CVPR*.
