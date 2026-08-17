# AI-Generated Image Detection

Detect whether an image is authentic or AI-generated, then see **why** with a Grad-CAM overlay.

An EfficientNet-B3 detector trained on CIFAKE (Stable Diffusion v1.4) is evaluated on later generators — StyleGAN, SD3/Flux, Midjourney v6, GPT-4o, and Janus-Pro — to measure how well detection generalises.

`[Python 3.10+]` `[PyTorch]` `[EfficientNet-B3]` `[Grad-CAM]` `[Gradio]`

**EECS 4080** · York University · Krishi Rajeshkumar Shah (220968905) · Supervisor: Mona Nasery · Summer 2026

---

## Demo

The Gradio app classifies an image as REAL or FAKE, reports temperature-scaled confidence, and highlights the regions that drove the decision.

![Gradio demo of AI image detection](assets/demo.mp4)

```bash
git clone https://github.com/krishi-shah/ai-image-detection.git
cd ai-image-detection
pip install -r requirements.txt
```

1. Download [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) into `data/raw/cifake/` (`train/` and `test/` with `REAL/` and `FAKE/`).
2. Place the checkpoint at `outputs/checkpoints/best_detector.pth`.
3. Launch the demo:

```bash
python app.py                 # http://localhost:7860
python app.py --share         # public gradio.live URL
```

Full setup (Colab, datasets, tests): **[`docs/SETUP.md`](docs/SETUP.md)**

---

## Research Question

> How well do AI-generated image detectors trained on current benchmark datasets generalise to images produced by next-generation generative models, and what accounts for the performance gap?

---

## Key Results

| Metric | CIFAKE (in-distribution) | Cross-generator |
|--------|--------------------------|-----------------|
| Accuracy | **96.96%** | GPT-4o 92.33%; others ≥96.17% |
| AUC-ROC | **0.9971** | ≥0.9926 on all families |
| Fake detection rate | 95.29% | GPT-4o **86.33%**; others 94–97% |
| ECE (pre-calibration) | 0.0026 | Rises under distribution shift |
| Learned temperature | 1.2189 | Does not transfer OOD |

### CIFAKE test set

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| FAKE | 98.58% | 95.29% | 96.91% | 10,000 |
| REAL | 95.44% | 98.63% | 97.01% | 10,000 |

### Cross-generator

| Generator | Accuracy | Fake detection rate | Acc. drop (rel) |
|-----------|----------|---------------------|-----------------|
| CIFAKE baseline | 0.9696 | — | 0% |
| GPT-4o | **0.9233** | **0.8633** | **4.8%** |
| Janus-Pro | 0.9617 | 0.9400 | 0.8% |
| Midjourney v6 | 0.9617 | 0.9400 | 0.8% |
| SD3/Flux | 0.9767 | 0.9700 | −0.7% |
| StyleGAN | 0.9617 | 0.9400 | 0.8% |

**Takeaways**

- In-distribution performance is near-perfect (96.96% accuracy, AUC 0.9971).
- Transfer is strong (94–97% fake detection) except on **GPT-4o** (86.3%).
- Grad-CAM, FFT, and t-SNE support a GPT-4o **feature-absence** story: later models leave fewer of the artifacts this detector learned on CIFAKE.
- Temperature scaling fitted on CIFAKE **worsens ECE** under distribution shift.

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

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_eda.ipynb` | Environment + CIFAKE EDA |
| `02_baseline_training.ipynb` | Hyperparameter sweep + training |
| `03_calibration.ipynb` | Temperature scaling, threshold analysis |
| `04_gradcam.ipynb` | Grad-CAM heatmaps |
| `05_generalisation_eval.ipynb` | Cross-generator evaluation |
| `06_gpt4o_investigation.ipynb` | FFT, t-SNE, Grad-CAM metrics |

Colab: open any notebook → Runtime → T4 GPU.

---

## Tests

```bash
# Avoid broken third-party pytest plugins on some machines
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'   # PowerShell
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1   # bash

pytest tests/ -v
```

---

## References

- Bird, J.J. & Lotfi, A. (2024). CIFAKE… *IEEE Access*.
- Corvi, R. et al. (2023). Detection of Diffusion Synthetic Images. *ICASSP*.
- Gragnaniello, D. et al. (2021). Are GAN Generated Images Easy to Detect? *ICME*.
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
- Rossler, A. et al. (2019). FaceForensics++. *ICCV*.
- Selvaraju, R. et al. (2017). Grad-CAM. *ICCV*.
- Wang, S. et al. (2020). CNN-Generated Images Are Surprisingly Easy to Spot… *CVPR*.
