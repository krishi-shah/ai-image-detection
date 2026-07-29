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

if BASELINE_PATH.exists():
    with open(BASELINE_PATH) as f:
        _baseline = json.load(f)
    TEMPERATURE = _baseline.get("temperature", 1.0)

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
# Theme
# ---------------------------------------------------------------------------

class DetectorTheme(gr.themes.Soft):
    def __init__(self):
        super().__init__(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.gray,
            text_size=gr.themes.sizes.text_lg,
            spacing_size=gr.themes.sizes.spacing_lg,
            radius_size=gr.themes.sizes.radius_lg,
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        )
        self.set(
            body_background_fill="#f9fafb",
            body_background_fill_dark="#f9fafb",
            block_background_fill="white",
            block_background_fill_dark="white",
            block_border_width="1px",
            block_border_color="#e5e7eb",
            block_border_color_dark="#e5e7eb",
            block_shadow="0 2px 6px rgba(0,0,0,0.06)",
            block_shadow_dark="0 2px 6px rgba(0,0,0,0.06)",
            block_radius="14px",
            block_label_text_size="*text_md",
            block_label_text_color="*neutral_700",
            block_label_text_color_dark="*neutral_700",
            block_title_text_color="*neutral_800",
            block_title_text_color_dark="*neutral_800",
            body_text_color="*neutral_800",
            body_text_color_dark="*neutral_800",
            body_text_color_subdued="*neutral_500",
            body_text_color_subdued_dark="*neutral_500",
            background_fill_primary="white",
            background_fill_primary_dark="white",
            background_fill_secondary="#f3f4f6",
            background_fill_secondary_dark="#f3f4f6",
            border_color_primary="#e5e7eb",
            border_color_primary_dark="#e5e7eb",
            color_accent_soft="*primary_50",
            color_accent_soft_dark="*primary_50",
            button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
            button_primary_background_fill_hover="linear-gradient(135deg, #4338ca, #6d28d9)",
            button_primary_background_fill_dark="linear-gradient(135deg, #4f46e5, #7c3aed)",
            button_primary_background_fill_hover_dark="linear-gradient(135deg, #4338ca, #6d28d9)",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            button_primary_border_color="transparent",
            button_primary_shadow="0 4px 12px rgba(79,70,229,0.35)",
            button_large_text_size="*text_lg",
            button_large_padding="14px 28px",
            input_background_fill="white",
            input_background_fill_dark="white",
            input_border_color="#d1d5db",
            input_border_color_dark="#d1d5db",
            input_border_width="1.5px",
            input_radius="12px",
        )


CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    padding: 20px 40px !important;
}
.verdict-fake {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border-left: 6px solid #dc2626;
    padding: 24px;
    border-radius: 14px;
    margin-bottom: 16px;
}
.verdict-real {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-left: 6px solid #16a34a;
    padding: 24px;
    border-radius: 14px;
    margin-bottom: 16px;
}
.verdict-title {
    font-size: 1.8em;
    font-weight: 700;
    margin: 0 0 8px 0;
}
.verdict-subtitle {
    color: #64748b;
    font-size: 1.1em;
}
.conf-bar-wrap {
    margin: 20px 0 12px;
}
.conf-bar {
    display: flex;
    height: 40px;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    font-size: 15px;
    font-weight: 700;
    line-height: 40px;
}
.conf-fake {
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    text-align: center;
}
.conf-real {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white;
    text-align: center;
}
.conf-labels {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #94a3b8;
    margin-top: 6px;
    padding: 0 4px;
}
.low-conf-note {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-left: 5px solid #d97706;
    padding: 14px 18px;
    border-radius: 12px;
    margin-top: 16px;
    font-size: 1em;
    color: #92400e;
}
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin: 18px 0;
}
.stat-card {
    text-align: center;
    padding: 18px 10px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    background: linear-gradient(180deg, #ffffff, #f8fafc);
}
.stat-val {
    font-size: 1.7em;
    font-weight: 700;
}
.stat-lbl {
    font-size: 0.82em;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.attention-tag {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 24px;
    font-size: 1em;
    font-weight: 600;
}
.att-focused { background: #dcfce7; color: #166534; }
.att-moderate { background: #fef3c7; color: #92400e; }
.att-diffuse { background: #fee2e2; color: #991b1b; }
.info-note {
    margin-top: 16px;
    padding: 14px 18px;
    background: #f1f5f9;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    font-size: 0.92em;
    color: #64748b;
}
.pipeline-bar {
    display: flex;
    gap: 0;
    margin: 0 0 20px 0;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    background: white;
}
.pipeline-step {
    flex: 1;
    text-align: center;
    padding: 18px 10px;
    border-right: 1px solid #e2e8f0;
}
.pipeline-step:last-child { border-right: none; }
.pipe-num {
    display: inline-block;
    width: 30px;
    height: 30px;
    line-height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 6px;
}
.pipe-title { font-size: 1em; font-weight: 600; color: #1e293b; }
.pipe-desc { font-size: 0.82em; color: #94a3b8; margin-top: 2px; }
.footer {
    text-align: center;
    padding: 20px;
    margin-top: 16px;
    color: #94a3b8;
    font-size: 0.88em;
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attention_info(heatmap: np.ndarray) -> tuple[str, str, str]:
    """Return (description, css_class, tag_text) for the attention pattern."""
    flat = heatmap.flatten()
    n = max(len(flat) // 10, 1)
    top10_mass = float(np.partition(flat, -n)[-n:].sum() / (flat.sum() + 1e-10))

    if top10_mass > 0.5:
        return ("Strong artifact signal detected", "att-focused", "Focused")
    elif top10_mass > 0.3:
        return ("Partial artifact signal", "att-moderate", "Moderate")
    else:
        return ("No clear artifact signal", "att-diffuse", "Diffuse")


def _build_verdict_html(pred_idx: int, max_conf: float,
                        p_fake: float, p_real: float,
                        attention_desc: str, att_class: str,
                        att_tag: str, temperature: float) -> str:
    """Build the full results HTML panel."""

    if pred_idx == 0:
        card_cls = "verdict-fake"
        title = "AI-Generated (FAKE)"
    else:
        card_cls = "verdict-real"
        title = "Authentic (REAL)"

    if max_conf >= 0.9:
        tier = "High confidence"
    elif max_conf >= 0.7:
        tier = "Moderate confidence"
    else:
        tier = "Low confidence"

    html = (
        f"<div class='{card_cls}'>"
        f"<div class='verdict-title'>{title}</div>"
        f"<div class='verdict-subtitle'>{tier} &mdash; {max_conf:.1%}</div>"
        f"</div>"
    )

    fake_pct = max(int(p_fake * 100), 3)
    real_pct = max(100 - fake_pct, 3)
    html += (
        "<div class='conf-bar-wrap'>"
        "<div class='conf-bar'>"
        f"<div class='conf-fake' style='width:{fake_pct}%'>{p_fake:.0%}</div>"
        f"<div class='conf-real' style='width:{real_pct}%'>{p_real:.0%}</div>"
        "</div>"
        "<div class='conf-labels'><span>AI-Generated</span><span>Authentic</span></div>"
        "</div>"
    )

    if max_conf < 0.7:
        html += (
            "<div class='low-conf-note'>"
            "Low confidence may indicate an image from a generator or domain "
            "the model was not trained on."
            "</div>"
        )

    html += (
        "<div class='stats-grid'>"
        f"<div class='stat-card'>"
        f"<div class='stat-val' style='color:#ef4444'>{p_fake:.1%}</div>"
        f"<div class='stat-lbl'>P(Fake)</div></div>"
        f"<div class='stat-card'>"
        f"<div class='stat-val' style='color:#22c55e'>{p_real:.1%}</div>"
        f"<div class='stat-lbl'>P(Real)</div></div>"
        f"<div class='stat-card'>"
        f"<div class='stat-val' style='color:#6366f1'>T={temperature:.2f}</div>"
        f"<div class='stat-lbl'>Calibration</div></div>"
        "</div>"
    )

    html += (
        f"<div style='text-align:center; margin:12px 0;'>"
        f"<span class='attention-tag {att_class}'>"
        f"Grad-CAM: {att_tag}</span>"
        f"<div style='font-size:0.9em; color:#94a3b8; margin-top:6px;'>"
        f"{attention_desc}</div></div>"
    )

    html += (
        "<div class='info-note'>"
        "<strong>Note:</strong> Trained on CIFAKE (Stable Diffusion v1.4). "
        "Performance may degrade on newer generators (GPT-4o, Midjourney v6). "
        "Confidence calibrated via temperature scaling (Guo et al., 2017)."
        "</div>"
    )

    return html


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def analyse_image(
    image_input,
    model=None,
    device=None,
    temperature=None,
    transform=None,
) -> tuple:
    """Run detection, calibration, and Grad-CAM on a single image.

    Args:
        image_input: numpy array (from Gradio) or PIL Image.

    Returns:
        (verdict_html, confidence_dict, original_resized, overlay_image, details_md)
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
            "## Model not loaded\n\n"
            f"Place the trained checkpoint at "
            f"`{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}` and restart.\n\n"
            "Run notebook `02_baseline_training.ipynb` to train the model."
        )
        return error_msg, {}, None, None, ""

    if isinstance(image_input, str):
        pil_image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        pil_image = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_image = image_input
    else:
        pil_image = image_input

    pil_image = pil_image.convert("RGB")

    resized = pil_image.resize((224, 224), Image.LANCZOS)
    original_np = np.array(resized).astype(np.float32) / 255.0

    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
    logits_np = logits.cpu().numpy()

    probs = apply_temperature(logits_np, temperature)
    p_fake = float(probs[0, 0])
    p_real = float(probs[0, 1])
    pred_idx = int(np.argmax(probs[0]))

    confidence_dict = {"AI-Generated": p_fake, "Authentic": p_real}

    heatmap = run_gradcam(model, tensor, model.conv_head)
    overlay = show_cam_on_image(original_np, heatmap, use_rgb=True)
    original_display = (original_np * 255).astype(np.uint8)

    max_conf = max(p_fake, p_real)
    attention_desc, att_class, att_tag = _attention_info(heatmap)

    verdict_html = _build_verdict_html(
        pred_idx, max_conf, p_fake, p_real,
        attention_desc, att_class, att_tag, temperature,
    )

    return verdict_html, confidence_dict, original_display, overlay, ""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""

    demo_samples_dir = PROJECT_ROOT / "data" / "demo_samples"
    examples = []
    if demo_samples_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            examples.extend(
                [str(p)] for p in sorted(demo_samples_dir.glob(ext))
            )

    with gr.Blocks(title="AI Image Detector") as demo:

        gr.Markdown(
            "# AI-Generated Image Detector\n\n"
            "Upload any image to check whether it was created by AI. "
            "Powered by **EfficientNet-B3** with calibrated confidence "
            "and **Grad-CAM** visual explanations."
        )

        gr.HTML(
            "<div class='pipeline-bar'>"
            "<div class='pipeline-step'>"
            "<div class='pipe-num'>1</div>"
            "<div class='pipe-title'>Upload</div>"
            "<div class='pipe-desc'>Any image</div></div>"
            "<div class='pipeline-step'>"
            "<div class='pipe-num'>2</div>"
            "<div class='pipe-title'>Detect</div>"
            "<div class='pipe-desc'>EfficientNet-B3</div></div>"
            "<div class='pipeline-step'>"
            "<div class='pipe-num'>3</div>"
            "<div class='pipe-title'>Calibrate</div>"
            "<div class='pipe-desc'>T = 1.22</div></div>"
            "<div class='pipeline-step'>"
            "<div class='pipe-num'>4</div>"
            "<div class='pipe-title'>Explain</div>"
            "<div class='pipe-desc'>Grad-CAM</div></div>"
            "</div>"
        )

        if not MODEL_LOADED:
            gr.Markdown(
                "> **Warning:** No model checkpoint found. "
                "The demo will not produce predictions until a trained "
                "checkpoint is available."
            )

        # --- Input section ---
        gr.Markdown("## Upload an Image")
        with gr.Row():
            image_input = gr.Image(
                type="pil",
                label="Drop an image or click to upload",
            )

        analyse_btn = gr.Button(
            "Analyse Image",
            variant="primary",
            size="lg",
        )

        # --- Examples ---
        if examples:
            gr.Examples(
                examples=examples,
                inputs=image_input,
                label="Or try these sample images",
            )

        # --- Results section ---
        gr.Markdown("---")
        gr.Markdown("## Results")

        verdict_output = gr.HTML(
            value=(
                "<div style='text-align:center; padding:40px 20px; color:#94a3b8;'>"
                "<div style='font-size:1.2em;'>Results will appear here after analysis</div>"
                "</div>"
            ),
        )

        with gr.Row():
            original_output = gr.Image(
                label="Original (224 x 224)",
                type="numpy",
            )
            heatmap_output = gr.Image(
                label="Grad-CAM Heatmap",
                type="numpy",
            )

        # Hidden outputs for function signature compatibility
        confidence_output = gr.Label(visible=False)
        details_output = gr.Markdown(visible=False)

        analyse_btn.click(
            fn=analyse_image,
            inputs=[image_input],
            outputs=[verdict_output, confidence_output,
                     original_output, heatmap_output, details_output],
        )

        # --- Info sections ---
        gr.Markdown("---")

        with gr.Accordion("How It Works", open=False):
            gr.Markdown(
                "### Detection Pipeline\n\n"
                "1. **Detection:** EfficientNet-B3 fine-tuned on 60,000 CIFAKE images "
                "(96.96% test accuracy, AUC 0.9971).\n"
                "2. **Calibration:** Post-hoc temperature scaling (T=1.2189) adjusts "
                "confidence to better reflect true accuracy (Guo et al., 2017).\n"
                "3. **Explainability:** Grad-CAM highlights the most influential image "
                "regions (Selvaraju et al., 2017).\n\n"
                "### What to Look For\n\n"
                "| Image Source | Expected Confidence | Grad-CAM Pattern |\n"
                "|---|---|---|\n"
                "| CIFAKE (SD v1.4) | High (~95%+) | Focused on artifact regions |\n"
                "| Midjourney v6 | Low-Moderate | Moderately diffuse |\n"
                "| GPT-4o | Low (~50%) | Fully diffuse, no clear signal |\n\n"
                "This demonstrates the **generalisation gap**: the detector relies on "
                "Stable Diffusion-specific artifacts that newer models don't produce."
            )

        with gr.Accordion("About This Project", open=False):
            gr.Markdown(
                "**Project:** AI-Generated Image Detection -- Evaluating Generalisation "
                "Across Generative Model Generations\n\n"
                "**Course:** EECS 4080 -- Computer Science Project, York University\n\n"
                "**Author:** Krishi Rajeshkumar Shah\n\n"
                "**Supervisor:** Mona Nasery\n\n"
                "**Research Question:** How well do AI-generated image detectors trained "
                "on current benchmark datasets generalise to images produced by "
                "next-generation generative models?\n\n"
                "**Key Finding:** Cross-generator evaluation shows significant performance "
                "degradation on newer generators (GPT-4o, Midjourney v6), confirming the "
                "detector relies on SD-specific artifacts that newer models no longer produce."
            )

        gr.HTML(
            "<div class='footer'>"
            "EECS 4080 &bull; York University &bull; "
            "EfficientNet-B3 + Temperature Scaling + Grad-CAM"
            "</div>"
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
        theme=DetectorTheme(),
        css=CUSTOM_CSS,
    )
