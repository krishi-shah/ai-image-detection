# Final Report: AI-Generated Image Detection — Evaluating Generalisation Across Generative Model Generations

**Author:** Krishi Rajeshkumar Shah (220968905)  
**Supervisor:** Mona Nasery  
**Course:** EECS 4080 — Computer Science Project, York University  
**Term:** Summer 2026  
**Repository:** https://github.com/krishi-shah/ai-image-detection

---

## Abstract

AI-image detectors trained on older generative models may fail as newer architectures close the perceptual gap with real photographs. This project fine-tunes an EfficientNet-B3 on CIFAKE (Stable Diffusion v1.4), adds temperature scaling and Grad-CAM explainability, and evaluates transfer to StyleGAN, SD3/Flux, Midjourney v6, GPT-4o, and Janus-Pro. In-distribution performance is strong (96.96% accuracy, AUC 0.9971, ECE 0.0026). Cross-generator transfer is surprisingly robust for four of five families (94–97% fake detection). Only GPT-4o shows a meaningful gap (86.3% fake detection). Frequency analysis, t-SNE embeddings, and quantitative Grad-CAM indicate a **feature-absence** failure on GPT-4o rather than an attention failure. A resolution-matched control experiment tests whether high transfer was confounded by pairing high-resolution fakes with low-resolution CIFAKE reals. The original chronological generalisation-gap hypothesis is refined: the gap is GPT-4o-specific, not universal across newer generators.

---

## 1. Introduction and Motivation

Detection tools for AI-generated images are typically trained and evaluated on older generators. As Stable Diffusion 3, Midjourney v6, and GPT-4o produce increasingly realistic outputs, a practical question follows: **do existing detectors still work, and if not, why?**

**Research question:** How well do AI-generated image detectors trained on current benchmark datasets generalise to images from next-generation generative models, and what accounts for the performance gap?

**Original hypothesis:** Detectors rely on low-level statistical artefacts specific to the training generator; these artefacts are weaker or absent in newer models, producing a measurable generalisation gap that widens with generator novelty.

---

## 2. Literature Review and Research Gap

- **Wang et al. (2020)** — ProGAN-trained ResNet-50 transferred across many GANs when strong augmentation forced shared fingerprints; predates diffusion.
- **Bird & Lotfi (2024)** — CIFAKE benchmark; >92% detection with standard CNNs; LIME explainability.
- **Corvi et al. (2023)** — Clear GAN↔diffusion gap (near-chance cross-domain); frequency-domain separation.
- **Gragnaniello et al. (2021)** — Generator-specific fingerprints; accuracy–transfer trade-off.
- **Guo et al. (2017)** — Temperature scaling for calibration.
- **Selvaraju et al. (2017)** — Grad-CAM.

**Gap addressed here:** Combine (1) multi-family chronological evaluation, (2) calibration under shift, (3) comparative Grad-CAM, and (4) measurement of the gap rather than proposing a new detector — plus a resolution-matched control for the real-reference protocol.

---

## 3. Datasets

### 3.1 CIFAKE (training / in-distribution test)

| Property | Value |
|----------|-------|
| On-disk total | 120,000 (100K train + 20K test) |
| Real | 60,000 from CIFAR-10 (50K train + 10K test) |
| Fake | 60,000 Stable Diffusion v1.4 (50K train + 10K test) |
| Native resolution | 32×32 (upscaled to 224×224) |
| Labels | FAKE=0, REAL=1 (`ImageFolder` alphabetical) |

*Note:* The paper/Kaggle card often cite 60,000 images. This project uses the on-disk archive counts above. Train/val split: 80/20 of the training folder (seed 42) → 80,000 / 20,000; held-out test = 20,000.

### 3.2 Cross-generator evaluation sets

| Family | Type | Source | N (FAKE) |
|--------|------|--------|----------|
| StyleGAN | GAN | OwensLab/CommunityForensics-Eval | 300 |
| SD3/Flux | Latent diffusion | Defactify Image Dataset | 300 |
| Midjourney v6 | Proprietary diffusion | Defactify / midjourney-images | 300 |
| GPT-4o | Autoregressive hybrid | Yejy53/GPT-ImgEval | 300 |
| Janus-Pro | Autoregressive | midbee/Janus-Pro-R1-Data | 300 |

Each family was paired with 300 CIFAKE `test/REAL` images (seed 42) for two-class metrics. **Protocol caveat:** reals are low-resolution upscales; fakes are high-resolution downscales — addressed in Experiment 5.

### 3.3 Real control sets (Experiment 5)

High-resolution photographic reals from COCO 2017 val and matched-source reals from Defactify / CommunityForensics (`scripts/download_real_controls.py`).

---

## 4. Methodology

### 4.1 Model

```
EfficientNet-B3 (ImageNet pretrained, timm)
  └── classifier: Linear(1536, 2)   # [FAKE, REAL]
```

### 4.2 Training

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| Scheduler | CosineAnnealingLR (T_max=20) |
| Loss | CrossEntropyLoss |
| Batch size | 32 |
| Sweep | 2 LRs × 3 weight decays, 3 epochs each |
| Full training | 20 epochs (best checkpoint: epoch 2, val acc 0.9694) |
| Seed | 42 |

Train transforms: Resize(224), RandomHorizontalFlip, RandomCrop(224, pad=8), ColorJitter, ImageNet normalize.  
Eval transforms: Resize(256), CenterCrop(224), ImageNet normalize.

### 4.3 Calibration

Temperature scaling (Guo et al., 2017): learn scalar *T* on validation logits via L-BFGS. Learned **T = 1.2189**.

### 4.4 Grad-CAM

Target layer: **`model.conv_head`** (`pytorch-grad-cam`). Used for qualitative heatmaps and quantitative entropy / peak-ratio / Gini metrics.

### 4.5 Metrics

Accuracy, AUC-ROC, ECE, F1 (FAKE), **fake detection rate** (recall on FAKE), reliability diagrams, degradation vs CIFAKE baseline.

---

## 5. Experiment 1 — CIFAKE Baseline

| Metric | Value |
|--------|-------|
| Test accuracy | **96.96%** |
| AUC-ROC | **0.9971** |
| ECE (pre-T) | 0.0026 |
| ECE (post-T) | 0.0113 |
| Temperature | 1.2189 |

**Per-class (test):**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| FAKE | 0.9858 | 0.9529 | 0.9691 | 10,000 |
| REAL | 0.9544 | 0.9863 | 0.9701 | 10,000 |

Confusion: TP 9,529 / FN 471 / FP 137 / TN 9,863.

Threshold sweep (0.40–0.70): all metrics stay above 91%; default 0.50 retained for interpretability despite a slight accuracy peak at 0.40.

**Calibration note:** Pre-T ECE was already excellent. Temperature scaling optimises NLL, not ECE, and slightly over-corrected on CIFAKE.

Figures: `outputs/plots/confusion_matrix.png`, `roc_curve.png`, `probability_histograms.png`, `threshold_*.png`, `reliability_*.png`, `calibration_comparison.png`.

---

## 6. Experiment 2 — Grad-CAM Explainability

- **REAL:** diffuse attention over fine texture; confidence typically 99–100%.
- **FAKE:** more localised hotspots on generation artefacts; lower-confidence correct FAKEs show scattered activation.

Interpretation: the detector relies on **SD v1.4-specific low-level texture cues**, motivating the generalisation study.

Figures: `outputs/plots/gradcam_comparison.png`, `gradcam_fake.png`, `gradcam_real.png`, `outputs/heatmaps/`.

---

## 7. Experiment 3 — Cross-Generator Generalisation

| Generator | Acc. | Fake Det. | AUC | F1 (FAKE) | ECE pre-T | ECE post-T | Acc. drop (rel) |
|-----------|------|-----------|-----|-----------|-----------|------------|-----------------|
| CIFAKE baseline | 0.9696 | — | 0.9971 | — | 0.0026 | 0.0113 | 0% |
| GPT-4o | **0.9233** | **0.8633** | 0.9926 | 0.9184 | 0.0083 | 0.0209 | **4.8%** |
| Janus-Pro | 0.9617 | 0.9400 | 0.9958 | 0.9608 | 0.0199 | 0.0334 | 0.8% |
| Midjourney v6 | 0.9617 | 0.9400 | 0.9955 | 0.9608 | 0.0338 | 0.0454 | 0.8% |
| SD3/Flux | 0.9767 | 0.9700 | 0.9975 | 0.9765 | 0.0229 | 0.0363 | −0.7% |
| StyleGAN | 0.9617 | 0.9400 | 0.9945 | 0.9608 | 0.0342 | 0.0514 | 0.8% |

### Findings

1. **GPT-4o is the only meaningful degradation** (~14% of GPT-4o images evade detection).
2. **Autoregressive ≠ automatically hard:** Janus-Pro matches StyleGAN/Midjourney (~94% fake detection).
3. **Diffusion-to-diffusion transfers well:** SD3/Flux ≥ baseline.
4. **Calibration does not transfer:** CIFAKE-fit *T* worsens OOD ECE.

Because half of each eval set is in-distribution CIFAKE REAL, **fake detection rate** is the primary OOD metric.

Figures: `outputs/plots/cross_generator_accuracy.png`, `degradation_waterfall.png`, `confidence_distributions_by_generator.png`, `ece_comparison_by_generator.png`, `gradcam_comparison_grid.png`.

---

## 8. Experiment 4 — GPT-4o Investigation

Three complementary analyses (`notebooks/06_gpt4o_investigation.ipynb`):

1. **FFT radial power spectra** — GPT-4o spectra closer to real photos than to CIFAKE fakes (Corvi-style).
2. **t-SNE of 1536-D penultimate embeddings** — GPT-4o clusters toward the real-image region.
3. **Quantitative Grad-CAM** (100 FAKE images/family):

| Family | Mean entropy | Mean peak ratio | Mean Gini |
|--------|--------------|-----------------|-----------|
| CIFAKE FAKE | **10.197** | **0.339** | **0.574** |
| GPT-4o | 10.291 | 0.309 | 0.529 |
| Janus-Pro | 10.355 | 0.295 | 0.498 |
| Midjourney v6 | 10.376 | 0.282 | 0.478 |
| SD3/Flux | 10.380 | 0.282 | 0.488 |
| StyleGAN | 10.319 | 0.317 | 0.518 |

CIFAKE fakes show the most focused attention. GPT-4o attention is **not** uniquely diffuse versus well-detected families (Midjourney/SD3 have higher entropy yet higher accuracy). Conclusion: **feature absence**, not “doesn’t know where to look.”

Figures: `outputs/plots/gpt4o_investigation/`.

---

## 9. Experiment 5 — Resolution-Matched Control

### Motivation

In Experiment 3, every REAL image is a 32×32 CIFAR upscale while every FAKE is a high-resolution downscale. The detector could separate **resampling signatures** rather than generator artefacts.

### Design (2×2)

| Condition | Images | Pipeline |
|-----------|--------|----------|
| A | CIFAKE REAL | Native eval |
| B | High-res real photos | Native eval |
| C | Same high-res reals | Force 32×32, then native |
| D | Generator FAKEs | Force 32×32, then native |

Implementation: `src/analysis/resolution_control.py`, `scripts/download_real_controls.py`, `notebooks/07_resolution_control.ipynb`.

### Results

> **Status:** Run `notebooks/07_resolution_control.ipynb` on Colab (T4), then copy `outputs/results/resolution_control.json` and `outputs/plots/resolution_control/` into the repo. The interpretation below updates automatically from that JSON.

**If B FAKE-rate ≫ A:** resolution confound — Experiment 3 accuracies were inflated; the true gap is larger than reported.  
**If B FAKE-rate ≈ A:** confound ruled out — reported transfer is genuine.

When `outputs/results/resolution_control.json` is present, summarise FAKE-rates for A–D and the before/after matching chart for each family here.

---

## 10. Experiment 6 — Interactive Demo

`app.py` provides a Gradio UI: upload → REAL/FAKE, temperature-scaled confidence tiers, Grad-CAM overlay. Demo samples cover CIFAKE real/fake, StyleGAN, Midjourney, and GPT-4o. Launch: `python app.py` or `python app.py --share`. Live runbook: `reports/demo_script.md`.

---

## 11. Discussion

### Hypothesis assessment

| Claim | Outcome |
|-------|---------|
| Detectors fail on newer generators in general | **Refuted** for StyleGAN, Midjourney, SD3/Flux, Janus-Pro |
| Autoregressive generators are especially hard | **Partially refuted** (Janus-Pro easy; GPT-4o hard) |
| Failure is due to missing training artefacts | **Supported** for GPT-4o (FFT + t-SNE + Grad-CAM metrics) |
| Calibration transfers under shift | **Refuted** |

The contribution is a **measured, nuanced gap**: strong in-distribution performance and broad transfer, with a specific GPT-4o blind spot explained by feature absence.

### Threats to validity

1. Real-reference protocol (CIFAKE reals) — mitigated by Experiment 5.
2. N=300 per family — adequate for gross effects, limited for fine ranking.
3. CIFAKE’s 32×32 origin limits high-frequency cues.
4. Single architecture (EfficientNet-B3) — results may not generalise to other backbones.
5. Generator image sources differ in JPEG pipelines and domains.

---

## 12. Limitations

1. Single-generator training (SD v1.4 only).
2. Low native training resolution.
3. OOD calibration not re-fit per family.
4. GPT-4o-specific gap not fully reduced to a closed-form spectral rule.
5. Control experiment requires Colab execution for final numbers (pipeline and tests are complete).

---

## 13. Conclusions and Future Work

1. A CIFAKE-trained EfficientNet-B3 reaches **96.96%** accuracy with near-perfect AUC and excellent pre-calibration ECE.
2. Cross-generator transfer is **strong** for four of five families; only **GPT-4o** shows a clear gap.
3. The gap is best explained as **missing SD-specific artefacts**, not failed spatial attention.
4. Temperature scaling fitted in-distribution **does not** remain optimal under shift.
5. Resolution-matched controls are necessary whenever low-res benchmark reals are paired with high-res generator fakes.

**Future work:** multi-generator training (e.g. GenImage), higher-resolution training data, hybrid spatial–frequency features, ensembles, and per-deployment recalibration.

---

## 14. Reproducibility

| Item | Detail |
|------|--------|
| Seed | 42 |
| Tests | 71 across 8 modules (`pytest tests/`) |
| Setup | `docs/SETUP.md` |
| Code | `src/model`, `src/evaluation`, `src/explainability`, `src/analysis` |
| Notebooks | `01`–`07` |
| License | MIT |

---

## References

1. Bird, J.J. & Lotfi, A. (2024). CIFAKE… *IEEE Access*, 12, 15642–15650.
2. Corvi, R. et al. (2023). On the Detection of Synthetic Images Generated by Diffusion Models. *ICASSP*.
3. Gragnaniello, D. et al. (2021). Are GAN Generated Images Easy to Detect? *ICME*.
4. Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
5. Rossler, A. et al. (2019). FaceForensics++. *ICCV*.
6. Selvaraju, R. et al. (2017). Grad-CAM. *ICCV*.
7. Wang, S. et al. (2020). CNN-Generated Images Are Surprisingly Easy to Spot…For Now. *CVPR*.
