## Experiment 5 Results (auto-folded from resolution_control.json)

| Condition | FAKE-rate | Mean P(FAKE) | Median P(FAKE) | N |
|-----------|-----------|--------------|----------------|---|
| A: CIFAKE REAL (native) | 1.7% | 0.0318 | 0.0004 | 300 |
| B: Hi-res REAL (native) | 93.0% | 0.8203 | 0.8839 | 300 |
| C: Hi-res REAL (→32×32) | 37.3% | 0.3979 | 0.3134 | 300 |
| D: gpt4o FAKE (→32×32) | 53.7% | 0.5270 | 0.5287 | 300 |
| D: janus_pro FAKE (→32×32) | 47.3% | 0.4902 | 0.4782 | 300 |
| D: midjourney_v6 FAKE (→32×32) | 34.7% | 0.3972 | 0.3485 | 300 |
| D: sd3_flux FAKE (→32×32) | 62.3% | 0.5950 | 0.6243 | 300 |
| D: stylegan FAKE (→32×32) | 38.3% | 0.4116 | 0.3591 | 300 |

**Confound suspected:** True

High false-positive rate on high-resolution reals suggests the detector partly relies on resolution/resampling cues.

Figures: `outputs/plots/resolution_control/`
