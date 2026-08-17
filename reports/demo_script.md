# Live demo runbook

The demo runs **inside the talk, after Experiment 4**, and takes about 75 seconds. The spoken
lines are in [`talk_script.md`](talk_script.md); this file is the operational checklist and
the longer runbook if Q&A asks for more.

**App:** `python app.py` → http://localhost:7860
**Q&A answers:** [`qa_prep.md`](qa_prep.md)

---

## Before the call

1. Confirm `outputs/checkpoints/best_detector.pth` exists.
2. Confirm temperature T = 1.2189 in `outputs/results/baseline_results.json`.
3. Launch the app and click **both** demo samples once. The first inference loads the model
   and is slow; you do not want that happening on screen.
4. Leave the app on a second browser tab. Do not share it yet.
5. Close everything else you would not want visible when you switch shares.

---

## The two clicks

Switch the Zoom share to the browser tab. Click two samples, then switch back to the deck.

### Click 1 — **CIFAKE Fake**

**Expect:** FAKE, high confidence, a tight localised heatmap.

**Say:** "A Stable Diffusion image. Label, calibrated confidence, and the heatmap — tight
and localised, on the artefact regions. Exactly the behaviour from Experiment 2."

### Click 2 — **GPT-4o**

**Expect:** noticeably lower confidence. Study-wide detection was 86.3%.

**Say:** "Confidence drops noticeably. And the heatmap is still sensible — the model is
attending to reasonable structure. It is looking. The evidence is not there."

Switch back to the deck. Do not click anything else unless asked.

---

## If it fails

Stay on the demo slide and keep talking — the figure beside the script makes the same point. Do
not troubleshoot on air. Every figure the demo would have shown is already in the deck:

| Slide | Shows |
|---|---|
| Demo | Heatmaps on real and generated CIFAKE images |
| Experiment 2 | The same, plus the generated images the model missed |
| Experiment 3 | Cross-generator results |
| Experiment 4 | Why GPT-4o |

---

## Longer runbook — only if Q&A asks for more

### CIFAKE Real

**Expect:** REAL, high confidence, attention diffuse across textures.
**Say:** "On CIFAKE reals the model is confident and its attention is broad."

### StyleGAN, then Midjourney

**Expect:** both FAKE at high confidence — 94% detection each.
**Say:** "Transfer looks strong on these families. GPT-4o is the exception."

### If they point at the confidence meters

**Say:** "Temperature scaling was fitted on CIFAKE, at 1.2189. It did not transfer —
calibration error rose on every new generator. Treat out-of-distribution confidence as a
soft signal, not a probability."
