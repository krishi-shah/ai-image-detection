# Live Demo Script

**Presentation:** EECS 4080 final demo  
**App launch:** `python app.py` → http://localhost:7860  
**Colab public link:** `python app.py --share`

Estimated live demo time: 4–6 minutes.

---

## Before you start

1. Confirm `outputs/checkpoints/best_detector.pth` exists.
2. Confirm `outputs/results/baseline_results.json` exists (temperature = 1.2189).
3. Launch the app and keep a browser tab ready.
4. Have `data/demo_samples/` open so you can click examples quickly.
5. Fallback: if the live app fails, use the Grad-CAM figures in `outputs/plots/` and the screenshots narrative below.

---

## Demo sequence

### Beat 1 — Frame the problem (30s)

> “Detectors trained on older generators often look excellent on their own test set. The question is whether they still work on Midjourney, SD3, and GPT-4o — and why.”

Show the README research question briefly, then switch to the Gradio UI.

### Beat 2 — CIFAKE real (in-distribution) (~45s)

Upload / click: `01_cifake_real.jpg`

**Expect:** REAL, high confidence (typically >95%), Grad-CAM diffuse across textures.

**Say:** “On CIFAKE reals the model is confident and attention is broad — looking at natural texture rather than a single hotspot.”

### Beat 3 — CIFAKE fake (in-distribution) (~45s)

Upload: `02_cifake_fake.jpg`

**Expect:** FAKE, high confidence, more localised Grad-CAM.

**Say:** “On Stable Diffusion v1.4 fakes it latches onto localised artefact regions. That is what we trained it to see.”

### Beat 4 — Cross-generator success (~60s)

Upload: `03_stylegan.png`, then `04_midjourney.png`

**Expect:** Both FAKE at high confidence for StyleGAN / Midjourney in our study (≈94% fake detection).

**Say:** “Surprisingly, transfer to StyleGAN and Midjourney is strong. Diffusion-to-diffusion and even GAN transfer worked far better than the original gap hypothesis predicted.”

### Beat 5 — GPT-4o failure case (~60s)

Upload: `05_gpt4o.png`

**Expect:** Often lower confidence, sometimes misclassified as REAL. Study-wide fake detection rate was **86.3%** (vs 95%+ on CIFAKE fakes).

**Say:** “GPT-4o is the only family with a meaningful drop. FFT and t-SNE show its frequency signature and embeddings sit closer to real photos. The model attends somewhere reasonable but finds no SD v1.4 artefacts — a feature-absence problem.”

### Beat 6 — Calibration caveat (~30s)

Point at the confidence / tier UI.

**Say:** “Temperature scaling was fit on CIFAKE (T=1.2189). Under distribution shift, ECE got worse — calibration does not transfer. Treat out-of-distribution confidence as a soft signal, not a calibrated probability.”

### Beat 7 — Control experiment (~45s)

Show the FAKE-rate bar chart from `outputs/plots/resolution_control/fake_rate_by_condition.png`.

**Say:** “We checked whether high accuracy was just a resolution confound. CIFAKE reals are 32×32 upscales; generator images are high-res downscales. High-res real photos are called FAKE 93% of the time. After forcing everything to 32×32, fake detection collapses to roughly 35–62%. The strong cross-generator numbers were largely resolution-driven.”

---

## Prepared answers

### “Isn’t your generalisation accuracy inflated by CIFAKE reals?”

Yes — and Experiment 5 proved it. Every family was scored as 300 generator fakes + 300 CIFAKE reals. Worse: 93% of high-resolution *real* photos are predicted FAKE under the native pipeline. After forcing images to 32×32, fake detection falls to ~35–62%. The honest story is a resolution confound plus a residual generalisation gap; fake detection rate under matching is the metric to cite.

### “Why did ECE rise after temperature scaling?”

Pre-calibration ECE on CIFAKE was already 0.0026. Temperature scaling optimises NLL, not ECE. On an already well-calibrated model it slightly over-corrected (ECE → 0.0113). Under OOD generators, the CIFAKE-fit T made ECE worse still.

### “Why did SD3/Flux beat the baseline?”

Within sampling noise / domain similarity: SD3/Flux still produces diffusion-like texture cues the CIFAKE (SD v1.4) detector recognises. Fake detection was 97.0%. We do not claim it is “better than in-distribution” in a strong statistical sense — it shows transfer, not a paradox.

### “Did you prove the generalisation gap?”

Yes, after correcting the protocol. The naive Experiment 3 numbers looked like “only GPT-4o fails.” Experiment 5 shows those numbers were resolution-confounded. Matched-resolution evaluation reveals a broad gap. GPT-4o’s feature-absence analysis remains a useful native-protocol diagnosis.

### “What is the Grad-CAM target layer?”

`model.conv_head` (EfficientNet-B3 final conv), via `pytorch-grad-cam`.

---

## Fallback if the app will not load

1. Show `outputs/plots/gradcam_comparison.png` and `outputs/plots/cross_generator_accuracy.png`.
2. Walk the generalisation table from the final report (Section on Experiment 3).
3. Open notebook 06 outputs under `outputs/plots/gpt4o_investigation/` for the “why GPT-4o” story.
