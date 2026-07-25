"""Gradio interactive demo for AI-generated image detection.

Combines the EfficientNet-B3 detector, temperature-scaled calibration,
and Grad-CAM explainability into a single web interface.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

import gradio as gr

from src.model.detector import build_detector, load_checkpoint
from src.evaluation.calibration import apply_temperature
from src.explainability.gradcam import run_gradcam
from src.utils.data_loader import get_transforms

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_grad_cam")

# ---------------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_detector.pth"
BASELINE_PATH = PROJECT_ROOT / "outputs" / "results" / "baseline_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "AI-Generated (FAKE)", 1: "Authentic (REAL)"}
TRANSFORM = get_transforms("test")

MODEL = None
MODEL_LOADED = False
TEMPERATURE = 1.0

# Load temperature from baseline results
if BASELINE_PATH.exists():
    with open(BASELINE_PATH) as f:
        _baseline = json.load(f)
    TEMPERATURE = _baseline.get("temperature", 1.0)

# Load model checkpoint
try:
    MODEL = build_detector(pretrained=False)
    load_checkpoint(MODEL, str(CHECKPOINT_PATH))
    MODEL.to(DEVICE).eval()
    MODEL_LOADED = True
except Exception as e:
    _load_error = str(e)
    MODEL = None
    MODEL_LOADED = False


# ---------------------------------------------------------------------------
# Inference pipeline (dependency-injected for testability)
# ---------------------------------------------------------------------------

def analyse_image(
    pil_image: Image.Image,
    model=None,
    device=None,
    temperature=None,
    transform=None,
) -> tuple:
    """Run detection, calibration, and Grad-CAM on a single image.

    Args:
        pil_image: Input image (any mode, any size).
        model: Detector model (defaults to global MODEL).
        device: Torch device (defaults to global DEVICE).
        temperature: Calibration temperature (defaults to global TEMPERATURE).
        transform: Preprocessing transform (defaults to global TRANSFORM).

    Returns:
        (verdict_html, confidence_dict, overlay_image, details_markdown)
    """
    if model is None:
        model = MODEL
    if device is None:
        device = DEVICE
    if temperature is None:
        temperature = TEMPERATURE
    if transform is None:
        transform = TRANSFORM

    _using_global = model is MODEL
    if model is None or (_using_global and not MODEL_LOADED):
        error_msg = (
            "**Model not loaded.** Place the trained checkpoint at "
            f"`{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}` and restart.\n\n"
            "Run notebook `02_baseline_training.ipynb` to train the model, "
            "or copy the checkpoint from Google Drive."
        )
        return error_msg, {}, None, ""

    pil_image = pil_image.convert("RGB")

    resized = pil_image.resize((224, 224), Image.LANCZOS)
    original_np = np.array(resized).astype(np.float32) / 255.0

    tensor = transform(pil_image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(tensor)
    logits_np = logits.cpu().numpy()

    # Temperature-scaled calibration
    probs = apply_temperature(logits_np, temperature)
    p_fake = float(probs[0, 0])
    p_real = float(probs[0, 1])
    pred_idx = int(np.argmax(probs[0]))

    confidence_dict = {"AI-Generated": p_fake, "Authentic": p_real}

    # Grad-CAM heatmap
    heatmap = run_gradcam(model, tensor, model.conv_head)
    overlay = show_cam_on_image(original_np, heatmap, use_rgb=True)

    # Verdict banner
    max_conf = max(p_fake, p_real)
    if max_conf >= 0.9:
        tier = "High confidence"
    elif max_conf >= 0.7:
        tier = "Moderate confidence"
    else:
        tier = "Low confidence \u2014 the model is uncertain about this image"

    if pred_idx == 0:
        verdict = (
            f"<div style='background:#fee2e2; border-left:4px solid #dc2626; "
            f"padding:12px; border-radius:4px; margin-bottom:8px;'>"
            f"<strong style='font-size:1.1em;'>\u26a0\ufe0f AI-Generated (FAKE)</strong><br>"
            f"<span style='color:#666;'>{tier} \u2014 {max_conf:.1%}</span></div>"
        )
    else:
        verdict = (
            f"<div style='background:#dcfce7; border-left:4px solid #16a34a; "
            f"padding:12px; border-radius:4px; margin-bottom:8px;'>"
            f"<strong style='font-size:1.1em;'>\u2705 Authentic (REAL)</strong><br>"
            f"<span style='color:#666;'>{tier} \u2014 {max_conf:.1%}</span></div>"
        )

    # Details panel
    details_lines = [
        f"**Calibrated confidence:** P(FAKE) = {p_fake:.1%}, P(REAL) = {p_real:.1%}",
        f"**Temperature:** T = {temperature:.4f}",
        "*Confidence scores calibrated via temperature scaling (Guo et al., 2017).*",
    ]
    if 0.35 < p_fake < 0.65:
        details_lines.append(
            "\n> **Note:** This image is near the decision boundary. "
            "The model cannot confidently classify it."
        )
    details_lines.append(
        "\n*This detector was trained on CIFAKE (Stable Diffusion v1.4 images). "
        "It may be less accurate on images from newer generators like GPT-4o "
        "or Midjourney v6.*"
    )
    details_md = "\n\n".join(details_lines)

    return verdict, confidence_dict, overlay, details_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks app (without launching)."""

    demo_samples_dir = PROJECT_ROOT / "data" / "demo_samples"
    examples = []
    if demo_samples_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            examples.extend(
                [str(p)] for p in sorted(demo_samples_dir.glob(ext))
            )

    with gr.Blocks(title="AI Image Detector") as demo:

        gr.Markdown(
            "# AI-Generated Image Detector\n"
            "Upload any image to check whether it is AI-generated or authentic. "
            "The system uses an EfficientNet-B3 classifier with calibrated confidence "
            "scores and Grad-CAM visual explanations."
        )

        if not MODEL_LOADED:
            gr.Markdown(
                "> **Warning:** No model checkpoint found at "
                f"`{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}`. "
                "The demo will not produce predictions until a trained "
                "checkpoint is available."
            )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    sources=["upload", "clipboard"],
                )
                analyse_btn = gr.Button("Analyse", variant="primary", size="lg")

                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=image_input,
                        label="Example Images",
                    )

            with gr.Column(scale=1):
                verdict_output = gr.Markdown(
                    value="*Upload an image and click Analyse to begin.*",
                    label="Verdict",
                )
                confidence_output = gr.Label(
                    label="Confidence Scores",
                    num_top_classes=2,
                )
                heatmap_output = gr.Image(
                    label="Grad-CAM Heatmap",
                    type="numpy",
                )
                details_output = gr.Markdown(label="Analysis Details")

        analyse_btn.click(
            fn=analyse_image,
            inputs=[image_input],
            outputs=[verdict_output, confidence_output, heatmap_output, details_output],
        )

        with gr.Accordion("How It Works", open=False):
            gr.Markdown(
                "1. **Detection:** EfficientNet-B3 fine-tuned on 60,000 CIFAKE images "
                "(96.96% test accuracy, AUC 0.9971). Classifies images as REAL or FAKE.\n"
                "2. **Calibration:** Post-hoc temperature scaling (T=1.2189) adjusts raw "
                "model confidence so scores better reflect true accuracy. Learned on a "
                "held-out validation set using L-BFGS (Guo et al., 2017).\n"
                "3. **Explainability:** Grad-CAM highlights which image regions most "
                "influenced the classification decision by visualising gradient-weighted "
                "activations from the last convolutional layer (Selvaraju et al., 2017)."
            )

        with gr.Accordion("About", open=False):
            gr.Markdown(
                "**Project:** AI-Generated Image Detection \u2014 Evaluating Generalisation "
                "Across Generative Model Generations\n\n"
                "**Course:** EECS 4080 \u2014 Computer Science Project, York University\n\n"
                "**Research Question:** How well do AI-generated image detectors trained on "
                "current benchmark datasets generalise to images produced by next-generation "
                "generative models, and what accounts for the performance gap?\n\n"
                "**Key Finding:** Cross-generator evaluation shows significant performance "
                "degradation on images from newer generators (GPT-4o, Midjourney v6), "
                "confirming the detector relies on Stable Diffusion-specific artifacts "
                "that newer models no longer produce."
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=("--share" in sys.argv),
        theme=gr.themes.Soft(),
    )
