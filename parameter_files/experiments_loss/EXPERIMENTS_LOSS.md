# GATES loss de-blurring experiments

Tracking doc for the `experiments_loss/` batch. Goal: test whether adding a
**gradient** or **structural** term to MSE reduces the blur that plain per-pixel
MSE causes (it rewards the conditional mean — see `BRIEF_A_loss_functions.md` for
the full rationale + literature).

Owner: Elena. Started 2026-09-23.

## Design

Every run is built from `experiments_tests/00_base_fixed.json` (fixed grid +
shuffle, SAHARA size 50, train 2014+2015 `freq=3`, test 2016 `freq=100`, `lr=5e-5`,
batch 5, 250 epochs, patience 100) and changes **only the training loss**. Test
loss is `MSELoss` (transformed space) for every run so results are comparable.

**Important — the control is plain MSE, not `experiments_tests/00`.** The new losses
have a plain (unweighted) MSE base with an additive term, so the clean A/B control
here is `L00` (plain `MSELoss`), *not* the `PixelWeightedMSELoss` used in the
capacity batch. `L00` therefore also doubles as a "does pixel-weighting even matter"
side-check against `experiments_tests/00`.

## Runs

| ID | File | Loss | vs L00 | Hypothesis |
|----|------|------|--------|------------|
| L00 | `L00_mse_control.json` | `MSELoss` | control (plain MSE) | baseline blur level |
| LG1 | `LG1_grad_b05.json` | `GradientMSELoss` β=0.5 | + gradient term (0.5×) | penalising ∇-error sharpens edges |
| LG2 | `LG2_grad_b10.json` | `GradientMSELoss` β=1.0 | + gradient term (1.0×) | how hard can we push sharpness before pixel accuracy drops |
| LG3 | `LG3_grad_charb_b05.json` | `GradientMSELoss` β=0.5, charbonnier | LG1 + robust core | L1-like core suits sparse/heavy-tailed fps |
| LS1 | `LS1_struct_g03.json` | `StructuralLoss` corr, γ=0.3 | + spatial-corr term (0.3×) | pattern-agreement pressure improves structure |
| LS2 | `LS2_struct_g10.json` | `StructuralLoss` corr, γ=1.0 | + spatial-corr term (1.0×) | stronger structural weight |

β/γ chosen so the added term is a minority of the loss at init; tune from the
results. `StructuralLoss` uses `mode="correlation"` (== ACC skill score);
`mode="ms_ssim"` is not implemented yet.

## How to run

```bash
python scripts/train_GATES_model.py --file_path parameter_files/experiments_loss/ L00_mse_control.json
# ...one per file; or loop the six in launch/launch_train.sh
```

## Results

Fill in as runs land. Report test `MSELoss` (transformed space) at best epoch, plus
a **blur/sharpness read-out** — the whole point of the batch is qualitative
sharpness, which test-MSE alone will *not* show (a sharper model may even score
slightly worse on MSE by the double-penalty argument).

| ID | Test MSE (best) | Best epoch | Sharper than L00? | Overfit? | Notes |
|----|-----------------|-----------|-------------------|----------|-------|
| L00 | | | — (reference) | | |
| LG1 | | | | | |
| LG2 | | | | | |
| LG3 | | | | | |
| LS1 | | | | | |
| LS2 | | | | | |

Suggested sharpness diagnostics (from the per-epoch `training_imgs/` and eval):
- eye-ball the prediction plots vs. truth (ridge/edge definition);
- gradient-magnitude ratio `mean|∇pred| / mean|∇target|` (blur → ≪ 1, ideal → ≈ 1);
- spatial correlation / ACC of pred vs. truth;
- power-spectrum / high-freq energy retained.

### Read-out guide
- **LG1/LG2 sharper, MSE ≈ L00** → gradient term works; pick the β that sharpens
  without hurting MSE, then combine with the best capacity/time config from
  `experiments_tests/`.
- **LG2 sharper but MSE worse than LG1** → sharpness↔accuracy trade-off; β=0.5 is
  the sweet spot.
- **LG3 ≈ LG1** → core choice doesn't matter here; keep squared. **LG3 > LG1** →
  robust core helps on the sparse tail; adopt charbonnier.
- **LS1/LS2 sharper** → structural pressure helps; correlation is cheap enough to
  keep. If it plateaus, that's the motivation to implement `mode="ms_ssim"`.
- **Nothing beats L00 on sharpness** → the blur is not a loss problem; pivot to the
  receptive-field / decoder workstreams (`BRIEF_B_static_mesh_skip.md`).

## Follow-ups
- Best gradient/structural term × best capacity+time config from `experiments_tests/`.
- Combine gradient **and** structural (both are additive) if each helps alone.
- Re-base the winning term on `PixelWeightedMSELoss` instead of plain MSE (the
  gradient/structural term is orthogonal to the pixel weighting).
- Implement `StructuralLoss` MS-SSIM path if correlation shows signal.
