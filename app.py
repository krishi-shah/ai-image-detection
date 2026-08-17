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
DEMO_SAMPLES_DIR = PROJECT_ROOT / "data" / "demo_samples"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "AI-Generated (FAKE)", 1: "Authentic (REAL)"}
TRANSFORM = get_transforms("test")
DISPLAY_MAX = 512
DISPLAY_MIN = 224

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


EXAMPLE_SPECS = [
    ("01_cifake_real.jpg", "CIFAKE Real"),
    ("02_cifake_fake.jpg", "CIFAKE Fake"),
    ("03_stylegan.png", "StyleGAN"),
    ("04_midjourney.png", "Midjourney"),
    ("05_gpt4o.png", "GPT-4o"),
]


def get_labeled_examples() -> list[tuple[str, str]]:
    """Return (path, label) pairs for demo samples that exist on disk."""
    pairs = []
    for filename, label in EXAMPLE_SPECS:
        path = DEMO_SAMPLES_DIR / filename
        if path.exists():
            pairs.append((str(path), label))
    return pairs


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

class DetectorTheme(gr.themes.Soft):
    def __init__(self):
        super().__init__(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            text_size=gr.themes.sizes.text_lg,
            spacing_size=gr.themes.sizes.spacing_md,
            radius_size=gr.themes.sizes.radius_lg,
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        )
        dark = "#0b1020"
        panel = "#12182b"
        self.set(
            body_background_fill=dark,
            body_background_fill_dark=dark,
            block_background_fill=panel,
            block_background_fill_dark=panel,
            block_border_width="1px",
            block_border_color="#1e293b",
            block_border_color_dark="#1e293b",
            block_shadow="0 8px 32px rgba(0,0,0,0.35)",
            block_shadow_dark="0 8px 32px rgba(0,0,0,0.35)",
            block_radius="16px",
            block_label_text_size="*text_sm",
            block_label_text_color="#94a3b8",
            block_label_text_color_dark="#94a3b8",
            block_title_text_color="#e2e8f0",
            block_title_text_color_dark="#e2e8f0",
            body_text_color="#e2e8f0",
            body_text_color_dark="#e2e8f0",
            body_text_color_subdued="#94a3b8",
            body_text_color_subdued_dark="#94a3b8",
            background_fill_primary=panel,
            background_fill_primary_dark=panel,
            background_fill_secondary="#0f1629",
            background_fill_secondary_dark="#0f1629",
            border_color_primary="#1e293b",
            border_color_primary_dark="#1e293b",
            color_accent_soft="#1e1b4b",
            color_accent_soft_dark="#1e1b4b",
            button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
            button_primary_background_fill_hover="linear-gradient(135deg, #6366f1, #8b5cf6)",
            button_primary_background_fill_dark="linear-gradient(135deg, #4f46e5, #7c3aed)",
            button_primary_background_fill_hover_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            button_primary_border_color="transparent",
            button_primary_shadow="0 4px 18px rgba(79,70,229,0.45)",
            button_large_text_size="*text_lg",
            button_large_padding="14px 28px",
            input_background_fill="#0f1629",
            input_background_fill_dark="#0f1629",
            input_border_color="#334155",
            input_border_color_dark="#334155",
            input_border_width="1.5px",
            input_radius="12px",
        )


CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 16px 24px 32px !important;
    background: #0b1020 !important;
}
.header-wrap {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 24px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}
.header-kicker {
    font-size: 0.75em;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #818cf8;
    font-weight: 600;
    margin-bottom: 6px;
}
.header-title {
    font-size: 2.05em;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    line-height: 1.15;
}
.header-sub {
    color: #94a3b8;
    margin-top: 6px;
    font-size: 0.98em;
}
.stat-strip {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.stat-pill {
    background: #12182b;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 10px 14px;
    min-width: 140px;
}
.stat-pill .num {
    font-size: 1.25em;
    font-weight: 700;
    color: #a5b4fc;
}
.stat-pill .lbl {
    font-size: 0.72em;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}
.verdict-fake {
    background: linear-gradient(135deg, #3f1219, #1a0b10);
    border: 1px solid #7f1d1d;
    border-left: 6px solid #ef4444;
    padding: 20px 22px;
    border-radius: 14px;
    margin-bottom: 12px;
}
.verdict-real {
    background: linear-gradient(135deg, #052e1c, #0b1020);
    border: 1px solid #14532d;
    border-left: 6px solid #22c55e;
    padding: 20px 22px;
    border-radius: 14px;
    margin-bottom: 12px;
}
.verdict-title {
    font-size: 2em;
    font-weight: 700;
    margin: 0 0 6px 0;
    color: #f8fafc;
}
.verdict-subtitle {
    color: #cbd5e1;
    font-size: 1.05em;
}
.conf-bar-wrap { margin: 16px 0 10px; }
.conf-bar {
    display: flex;
    height: 36px;
    border-radius: 10px;
    overflow: hidden;
    background: #0f1629;
    font-size: 14px;
    font-weight: 700;
    line-height: 36px;
}
.conf-fake {
    background: linear-gradient(135deg, #ef4444, #b91c1c);
    color: white;
    text-align: center;
}
.conf-real {
    background: linear-gradient(135deg, #22c55e, #15803d);
    color: white;
    text-align: center;
}
.conf-labels {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #64748b;
    margin-top: 6px;
}
.low-conf-note, .info-note {
    padding: 12px 16px;
    border-radius: 12px;
    margin-top: 12px;
    font-size: 0.92em;
}
.low-conf-note {
    background: #1c1408;
    border-left: 4px solid #d97706;
    color: #fde68a;
}
.info-note {
    background: #111827;
    border: 1px solid #1e293b;
    color: #94a3b8;
}
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin: 14px 0;
}
.stat-card {
    text-align: center;
    padding: 14px 8px;
    border-radius: 12px;
    border: 1px solid #1e293b;
    background: #0f1629;
}
.stat-val { font-size: 1.45em; font-weight: 700; }
.stat-lbl {
    font-size: 0.72em;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}
.attention-tag {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.92em;
    font-weight: 600;
}
.att-focused { background: #14532d; color: #bbf7d0; }
.att-moderate { background: #78350f; color: #fde68a; }
.att-diffuse { background: #7f1d1d; color: #fecaca; }
.empty-state {
    text-align: center;
    padding: 56px 24px;
    color: #64748b;
    border: 1px dashed #334155;
    border-radius: 16px;
    background: #0f1629;
}
.empty-state h3 { color: #e2e8f0; margin: 0 0 8px 0; }
.footer {
    text-align: center;
    padding: 18px 8px 8px;
    color: #64748b;
    font-size: 0.85em;
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_verdict_html() -> str:
    return (
        "<div class='empty-state'>"
        "<h3>Awaiting an image</h3>"
        "<div>Click a labeled sample on the left. "
        "Verdict, calibrated confidence, and Grad-CAM appear here.</div>"
        "</div>"
    )


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


def _science_note(pred_idx: int, max_conf: float, att_tag: str) -> str:
    """Contextual note aligned with Experiments 3–4."""
    if pred_idx == 0 and max_conf >= 0.9 and att_tag == "Focused":
        return (
            "High-confidence FAKE with focused Grad-CAM is typical of "
            "CIFAKE-style (Stable Diffusion v1.4) artifacts."
        )
    if max_conf < 0.7 or att_tag == "Diffuse":
        return (
            "Weak artifact signal — common for newer generators such as "
            "GPT-4o, where Experiment 4 found missing Stable Diffusion v1.4 "
            "fingerprints rather than unfocused attention."
        )
    return (
        "Trained on CIFAKE (SD v1.4). Cross-generator transfer is strong "
        "except GPT-4o (86.3% fake detection). "
        "Treat OOD confidence as a soft signal (T = 1.2189, fitted in-distribution)."
    )


def _fit_display(pil: Image.Image, max_side: int = DISPLAY_MAX,
                 min_side: int = DISPLAY_MIN) -> Image.Image:
    """Scale for projector visibility without exceeding max_side."""
    w, h = pil.size
    longest = max(w, h)
    shortest = min(w, h)
    if longest > max_side:
        scale = max_side / longest
    elif shortest < min_side:
        scale = min_side / shortest
        if longest * scale > max_side:
            scale = max_side / longest
    else:
        scale = 1.0
    if abs(scale - 1.0) < 1e-6:
        return pil
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil.resize((new_w, new_h), Image.LANCZOS)


def _display_pair(pil_image: Image.Image, heatmap_224: np.ndarray):
    """Original + Grad-CAM overlay at display resolution."""
    display_pil = _fit_display(pil_image)
    display_np = np.array(display_pil).astype(np.float32) / 255.0
    heat_img = Image.fromarray((np.clip(heatmap_224, 0, 1) * 255).astype(np.uint8))
    heat_up = np.array(
        heat_img.resize(display_pil.size, Image.BILINEAR)
    ).astype(np.float32) / 255.0
    overlay = show_cam_on_image(display_np, heat_up, use_rgb=True)
    original_display = (display_np * 255).astype(np.uint8)
    return original_display, overlay


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
        f"<div class='verdict-subtitle'>{tier} &mdash; {max_conf:.1%} calibrated</div>"
        f"</div>"
    )

    fake_pct = max(int(round(p_fake * 100)), 3)
    real_pct = max(100 - fake_pct, 3)
    html += (
        "<div class='conf-bar-wrap'>"
        "<div class='conf-bar'>"
        f"<div class='conf-fake' style='width:{fake_pct}%'>{p_fake:.0%}</div>"
        f"<div class='conf-real' style='width:{real_pct}%'>{p_real:.0%}</div>"
        "</div>"
        "<div class='conf-labels'><span>P(AI-Generated)</span>"
        "<span>P(Authentic)</span></div>"
        "</div>"
    )

    html += (
        "<div class='stats-grid'>"
        f"<div class='stat-card'>"
        f"<div class='stat-val' style='color:#a5b4fc'>{max_conf:.1%}</div>"
        f"<div class='stat-lbl'>Calibrated conf.</div></div>"
        f"<div class='stat-card'>"
        f"<div class='stat-val' style='color:#c4b5fd'>T={temperature:.2f}</div>"
        f"<div class='stat-lbl'>Temperature</div></div>"
        f"<div class='stat-card'>"
        f"<span class='attention-tag {att_class}'>{att_tag}</span>"
        f"<div class='stat-lbl'>Grad-CAM</div></div>"
        "</div>"
    )

    html += (
        f"<div style='text-align:center; margin:4px 0 8px;'>"
        f"<div style='font-size:0.9em; color:#94a3b8;'>{attention_desc}</div></div>"
    )

    if max_conf < 0.7:
        html += (
            "<div class='low-conf-note'>"
            "Low confidence: the model is uncertain — often a newer generator "
            "such as GPT-4o."
            "</div>"
        )

    html += f"<div class='info-note'><strong>Note:</strong> {_science_note(pred_idx, max_conf, att_tag)}</div>"
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

    Returns:
        (verdict_html, confidence_dict, original_display, overlay_image, details_md)
    """
    if model is None:
        model = MODEL
    if device is None:
        device = DEVICE
    if temperature is None:
        temperature = TEMPERATURE
    if transform is None:
        transform = TRANSFORM

    if image_input is None:
        return _empty_verdict_html(), {}, None, None, ""

    _using_global = model is MODEL
    if model is None or (_using_global and not MODEL_LOADED):
        error_msg = (
            "<div class='low-conf-note'><strong>Model not loaded.</strong> "
            f"Place the trained checkpoint at "
            f"<code>{CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}</code> and restart. "
            "Run notebook <code>02_baseline_training.ipynb</code> to train.</div>"
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
    original_display, overlay = _display_pair(pil_image, heatmap)

    max_conf = max(p_fake, p_real)
    attention_desc, att_class, att_tag = _attention_info(heatmap)

    verdict_html = _build_verdict_html(
        pred_idx, max_conf, p_fake, p_real,
        attention_desc, att_class, att_tag, temperature,
    )

    return verdict_html, confidence_dict, original_display, overlay, ""


def _load_example(path: str):
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""

    labeled = get_labeled_examples()
    example_buttons: list[tuple] = []

    with gr.Blocks(
        title="AI Image Detector",
        theme=DetectorTheme(),
        css=CUSTOM_CSS,
    ) as demo:

        gr.HTML(
            "<div class='header-wrap'>"
            "<div>"
            "<div class='header-kicker'>EECS 4080 · York University</div>"
            "<div class='header-title'>AI-Generated Image Detector</div>"
            "<div class='header-sub'>EfficientNet-B3 · temperature scaling "
            "(T = 1.2189) · Grad-CAM</div>"
            "</div>"
            "<div class='stat-strip'>"
            "<div class='stat-pill'><div class='num'>96.96%</div>"
            "<div class='lbl'>CIFAKE accuracy</div></div>"
            "<div class='stat-pill'><div class='num'>86.3%</div>"
            "<div class='lbl'>GPT-4o fake det.</div></div>"
            "<div class='stat-pill'><div class='num'>94%+</div>"
            "<div class='lbl'>Other-family fake det.</div></div>"
            "</div></div>"
        )

        if not MODEL_LOADED:
            gr.Markdown(
                "> **Warning:** No model checkpoint found. "
                "The demo will not produce predictions until a trained "
                "checkpoint is available at `outputs/checkpoints/best_detector.pth`."
            )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                image_input = gr.Image(
                    type="pil",
                    label="Selected sample",
                    height=420,
                    interactive=False,
                    sources=[],
                    buttons=[],
                    placeholder="Click a labeled sample below",
                )
                gr.Markdown(
                    "**Study samples** — click a labeled image from the evaluation set."
                )
                if labeled:
                    gr.Markdown("**One-click samples** — runs analysis immediately")
                    with gr.Row():
                        example_buttons = []
                        for path, label in labeled:
                            example_buttons.append(
                                (gr.Button(label, size="sm"), path)
                            )

            with gr.Column(scale=6):
                verdict_output = gr.HTML(value=_empty_verdict_html())
                with gr.Row():
                    original_output = gr.Image(
                        label="Original",
                        type="numpy",
                        height=280,
                        interactive=False,
                        sources=[],
                        buttons=[],
                    )
                    heatmap_output = gr.Image(
                        label="Grad-CAM",
                        type="numpy",
                        height=280,
                        interactive=False,
                        sources=[],
                        buttons=[],
                    )

        confidence_output = gr.Label(visible=False)
        details_output = gr.Markdown(visible=False)

        outputs = [
            verdict_output, confidence_output,
            original_output, heatmap_output, details_output,
        ]

        for btn, path in example_buttons:
            btn.click(
                fn=lambda p=path: _load_example(p),
                outputs=image_input,
            ).then(
                fn=analyse_image,
                inputs=[image_input],
                outputs=outputs,
            )

        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                "1. **Detect** — EfficientNet-B3 fine-tuned on CIFAKE "
                "(96.96% test accuracy, AUC 0.9971).\n"
                "2. **Calibrate** — temperature scaling, T = 1.2189 "
                "(Guo et al., 2017). Fitted on CIFAKE; ECE worsens under shift.\n"
                "3. **Explain** — Grad-CAM on `model.conv_head` "
                "(Selvaraju et al., 2017).\n\n"
                "**What to look for**\n\n"
                "| Sample | Typical result |\n"
                "|---|---|\n"
                "| CIFAKE Real | REAL, high confidence, diffuse CAM |\n"
                "| CIFAKE Fake | FAKE, high confidence, focused artifacts |\n"
                "| StyleGAN / Midjourney | Often FAKE, ~94% fake detection |\n"
                "| GPT-4o | Harder case (86.3% fake detection) |\n"
            )

        with gr.Accordion("Research findings", open=False):
            gr.Markdown(
                "StyleGAN / Midjourney / Janus-Pro ~94% fake detection; "
                "GPT-4o **86.3%**; SD3/Flux 97%. GPT-4o is the hard case: "
                "FFT, t-SNE, and Grad-CAM support feature absence. "
                "Temperature scaling fitted on CIFAKE does not transfer.\n\n"
                "Full write-up: `reports/final_report.md` · "
                "[GitHub](https://github.com/krishi-shah/ai-image-detection)"
            )

        with gr.Accordion("About this project", open=False):
            gr.Markdown(
                "**AI-Generated Image Detection — Evaluating Generalisation "
                "Across Generative Model Generations**\n\n"
                "EECS 4080 · York University · Krishi Rajeshkumar Shah · "
                "Supervisor: Mona Nasery\n\n"
                "**Research question:** How well do detectors trained on current "
                "benchmarks generalise to next-generation generators, and what "
                "accounts for the gap?"
            )

        gr.HTML(
            "<div class='footer'>"
            "EECS 4080 · York University · "
            "EfficientNet-B3 + temperature scaling + Grad-CAM"
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
