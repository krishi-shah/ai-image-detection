# Literature Review: AI-Generated Image Detection

**Project:** AI-Generated Image Detection — Evaluating Generalisation Across Generative Model Generations  
**Author:** Krishi Rajeshkumar Shah  
**Date:** May 2026

---

## 1. CNN-Based Detection Methods

The feasibility of detecting AI-generated images using convolutional neural networks was established by Wang et al. (2020), who demonstrated that a ResNet-50 classifier trained on ProGAN-generated images could generalise to images from other GAN architectures with surprisingly high accuracy. Their key finding was that simple data augmentation (Gaussian blur and JPEG compression) during training substantially improved cross-generator transfer. However, the study was conducted before the emergence of diffusion-based models, and all evaluation was limited to GAN-family outputs.

Bird and Lotfi (2024) introduced the CIFAKE dataset, comprising 60,000 real photographs from CIFAR-10 and 60,000 AI-generated counterparts produced using Stable Diffusion v1.4. Their work showed that standard convolutional classifiers could achieve over 90% accuracy on this specific distribution. Importantly, they also provided Explainable AI analysis using LIME to identify which input features drove classifier decisions — an approach this project extends with Grad-CAM.

Gragnaniello et al. (2021) provided a critical analysis of the state of the art in GAN detection, showing that detectors trained on one GAN architecture often failed on others, especially when architectural differences changed the low-level artifact profile. They demonstrated that spectral analysis could reveal GAN-specific fingerprints but that these fingerprints were not universal across model families — a finding directly relevant to this project's generalisation hypothesis.

**Synthesis:** Existing CNN-based detectors work well within the distribution they were trained on, but cross-generator generalisation remains fragile. No prior work has systematically measured this degradation across a chronological ladder of generative models spanning GANs, diffusion models, and autoregressive generators.

---

## 2. Evolution of Generative Models

Understanding why detection becomes harder over time requires understanding how generative architectures have evolved, since each generation produces fundamentally different statistical artifacts.

**Generative Adversarial Networks (GANs):** Early GANs (Goodfellow et al., 2014) and their successors — including ProGAN (Karras et al., 2018) and StyleGAN (Karras et al., 2019, 2020) — generate images by learning a mapping from a latent noise vector to pixel space through an adversarial training process. These models introduced characteristic artifacts: checkerboard patterns from transposed convolution upsampling, spectral peaks at specific frequencies, and colour distribution anomalies. These artifacts are precisely what early detectors learned to exploit.

**Diffusion Models:** Latent diffusion models, including Stable Diffusion (Rombach et al., 2022) and DALL-E 2/3 (Ramesh et al., 2022), generate images through an iterative denoising process in a learned latent space. This fundamentally different generation mechanism does not produce the transposed convolution artifacts that GANs do. Instead, diffusion outputs exhibit different statistical properties — smoother high-frequency spectra and fewer localised structural anomalies — which means detectors trained on GAN artifacts systematically fail.

**Autoregressive and Hybrid Models:** The latest generation — including GPT-4o (OpenAI, 2024) and Midjourney v6 (Midjourney, 2024) — uses autoregressive token prediction or proprietary hybrid architectures. These models produce images with even fewer detectable artifacts, having been trained at massive scale with extensive post-processing pipelines that further smooth out any residual statistical signatures.

**Key implication:** Each generation of models eliminates the specific artifacts that the previous generation's detectors relied upon, creating an escalating generalisation gap.

---

## 3. Visual Explainability: Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM), proposed by Selvaraju et al. (2017), provides visual explanations for CNN decisions by computing the gradient of the target class score with respect to the feature maps of a convolutional layer. These gradients are globally average-pooled to produce importance weights for each feature map channel, which are then used to generate a coarse localisation heatmap highlighting the image regions most relevant to the model's decision.

Grad-CAM is particularly appropriate for this project for two reasons. First, it is model-agnostic with respect to CNN architecture — it works with any convolutional network, including EfficientNet-B3. Second, it enables direct comparison of what the detector "looks at" when classifying images from different generative model generations. If the detector correctly identifies a GAN-generated image by attending to checkerboard artifacts in flat regions, but fails on a diffusion-generated image because those artifacts are absent, the Grad-CAM heatmaps will make this failure mode visually interpretable.

This visual interpretability is essential for answering not just whether the generalisation gap exists, but why it exists — which is the core analytical contribution of this project.

---

## 4. Confidence Calibration

Guo et al. (2017) demonstrated that modern deep neural networks are poorly calibrated: their softmax confidence scores tend to be overconfident, meaning a model that reports 95% confidence may only be correct 70% of the time. This miscalibration is problematic for any deployed detection system, since users need to know how much to trust a given prediction.

**Expected Calibration Error (ECE)** is the standard metric for measuring calibration quality. It partitions predictions into bins by confidence level and computes the weighted average of the absolute difference between accuracy and confidence within each bin. A perfectly calibrated model has ECE = 0.

**Temperature scaling** is a simple post-hoc calibration method that learns a single scalar parameter T on a held-out validation set. All logits are divided by T before applying softmax, which adjusts the confidence distribution without changing the predicted class. Guo et al. showed that temperature scaling is surprisingly effective despite its simplicity, often matching or outperforming more complex calibration methods.

For this project, calibration is important because the detector will encounter images from distributions it was not trained on (newer generative models). A calibrated model should ideally express lower confidence on out-of-distribution inputs rather than making high-confidence errors — providing an additional signal that the input may be from an unseen generator.

---

## 5. Research Gap

While individual components — CNN-based detection, Grad-CAM explainability, and calibration — have been studied in isolation, no prior work has combined them to systematically quantify and explain the performance degradation of a single detector across a chronological progression of generative model architectures. Corvi et al. (2023) showed that detectors trained on GAN outputs perform poorly on diffusion-generated images, confirming the existence of a generalisation gap, but their analysis did not extend to the newest autoregressive models (GPT-4o, Midjourney v6), did not incorporate calibration-aware evaluation (ECE), and did not use Grad-CAM to attribute specific failure modes to changes in the generative process.

This project addresses that gap by training a well-characterised baseline detector on CIFAKE, calibrating it with temperature scaling, and then systematically evaluating it — with both quantitative metrics (accuracy, AUC, ECE) and visual attribution (Grad-CAM) — across images from StyleGAN, Stable Diffusion, Midjourney v6, and GPT-4o. The contribution is not a new detection method, but a rigorous, explainable measurement of where and why current detection fails as generative technology advances.

---

## References

- Bird, J.J. & Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. *IEEE Access*, 12, 15642–15650.
- Corvi, R. et al. (2023). On the Detection of Synthetic Images Generated by Diffusion Models. *ICASSP 2023*.
- Goodfellow, I. et al. (2014). Generative Adversarial Nets. *NeurIPS 2014*.
- Gragnaniello, D. et al. (2021). Are GAN Generated Images Easy to Detect? A Critical Analysis of the State of the Art. *ICME 2021*.
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
- Karras, T. et al. (2018). Progressive Growing of GANs for Improved Quality, Stability, and Variation. *ICLR 2018*.
- Karras, T. et al. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. *CVPR 2019*.
- Ramesh, A. et al. (2022). Hierarchical Text-Conditional Image Generation with CLIP Latents. *arXiv:2204.06125*.
- Rombach, R. et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.
- Rossler, A. et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. *ICCV 2019*.
- Selvaraju, R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.
- Wang, S. et al. (2020). CNN-Generated Images Are Surprisingly Easy to Spot...For Now. *CVPR 2020*.
