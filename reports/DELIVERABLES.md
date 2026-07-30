# Deliverables Index

Maps each EECS 4080 contract deliverable to the concrete artifact in this repository.

| Contract deliverable | Artifact |
|----------------------|----------|
| Trained EfficientNet-B3 detector with calibrated confidence and Grad-CAM | `outputs/checkpoints/best_detector.pth`, `outputs/results/baseline_results.json` (T=1.2189), `src/explainability/gradcam.py`, `outputs/plots/gradcam_*.png`, `outputs/heatmaps/` |
| Structured generalisation evaluation across ≥4 generator families | `notebooks/05_generalisation_eval.ipynb`, `notebooks/06_gpt4o_investigation.ipynb`, `src/evaluation/generalisation.py`, `outputs/results/generalisation_results.json`, `outputs/results/degradation_summary.json`, `outputs/plots/cross_generator_*.png` |
| Interactive Gradio demo (classification + confidence + heatmap) | `app.py`, `tests/test_app.py`, `data/demo_samples/`, `reports/demo_script.md` |
| Final report (research question, methodology, experiments, conclusions) | `reports/final_report.md`, `reports/final_report.tex` |
| Documented source code, GitHub repo, user setup guide | https://github.com/krishi-shah/ai-image-detection , `README.md`, `docs/SETUP.md`, `LICENSE`, `requirements.txt` |
| Presentation | `reports/presentation.pptx` (built by `scripts/build_presentation.py`) |
| Progress report (Milestone 4) | `reports/progress_report.md`, `reports/progress_report.tex` |
| Literature review (Milestone 1) | `reports/literature_review.md` |
| Resolution-matched control (Milestone 6 addition) | `notebooks/07_resolution_control.ipynb`, `src/analysis/resolution_control.py`, `scripts/download_real_controls.py`, `outputs/results/resolution_control.json` |

## Evaluation criteria (from contract)

| Criterion | Weight | Where it is evidenced |
|-----------|--------|------------------------|
| Research quality & findings | 20% | Final report §§3–5, generalisation + GPT-4o + resolution control |
| Technical implementation | 20% | `src/`, notebooks 02–07, 71 tests |
| Presentation & demonstration | 30% | `reports/presentation.pptx`, `reports/demo_script.md`, live `app.py` |
| Final report | 20% | `reports/final_report.md` / `.tex` |
| Code quality & documentation | 10% | Modular `src/`, README, `docs/SETUP.md`, GitHub history |

## Submission checklist

- [ ] `reports/final_report.tex` compiles on Overleaf with figures from `outputs/`
- [ ] `reports/presentation.pptx` opens and embeds key figures
- [ ] `python app.py` runs locally with checkpoint
- [ ] `pytest tests/` — all green
- [x] Drive `outputs/` (Milestone 4 plots/JSONs) copied into the local repo if not already present
- [x] Notebook 07 run on Colab; `resolution_control.json` + plots committed or submitted
