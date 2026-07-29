"""Fold resolution_control.json into reports/_control_results_snippet.md when available.

Run after Colab notebook 07 outputs are copied into the repo:

    python scripts/fold_control_results.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "outputs" / "results" / "resolution_control.json"
OUT_MD = ROOT / "reports" / "_control_results_snippet.md"


def main() -> int:
    if not JSON_PATH.exists():
        OUT_MD.write_text(
            "<!-- Resolution control results pending. "
            "Run notebooks/07_resolution_control.ipynb and copy outputs/results/"
            "resolution_control.json into the repo, then re-run this script. -->\n",
            encoding="utf-8",
        )
        print(f"No results yet — wrote placeholder to {OUT_MD}")
        return 0

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    conds = data["conditions"]
    interp = data["interpretation"]

    lines = [
        "## Experiment 5 Results (auto-folded from resolution_control.json)",
        "",
        "| Condition | FAKE-rate | Mean P(FAKE) | Median P(FAKE) | N |",
        "|-----------|-----------|--------------|----------------|---|",
    ]
    for key, label in [
        ("A_cifake_real_native", "A: CIFAKE REAL (native)"),
        ("B_hires_real_native", "B: Hi-res REAL (native)"),
        ("C_hires_real_matched", "C: Hi-res REAL (→32×32)"),
    ]:
        c = conds[key]
        lines.append(
            f"| {label} | {c['fake_rate']:.1%} | {c['mean_p_fake']:.4f} | "
            f"{c['median_p_fake']:.4f} | {c['n_images']} |"
        )

    d = conds.get("D_generator_fakes_matched", {})
    for fam, c in sorted(d.items()):
        lines.append(
            f"| D: {fam} FAKE (→32×32) | {c['fake_rate']:.1%} | {c['mean_p_fake']:.4f} | "
            f"{c['median_p_fake']:.4f} | {c['n_images']} |"
        )

    lines += [
        "",
        f"**Confound suspected:** {interp.get('resolution_confound_suspected')}",
        "",
        interp.get("summary", ""),
        "",
        "Figures: `outputs/plots/resolution_control/`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
