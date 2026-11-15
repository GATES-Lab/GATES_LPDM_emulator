# How-To: Self-supervised met pretraining of the trunk

Pretrain the shared trunk (encoder + processor) of the dual model on meteorology alone, then
warm-start a dual run from it. Code: [`train_met_pretrain.py`](../train_met_pretrain.py) (entry
point), [`gates/training/pretraining.py`](../gates/training/pretraining.py) (pretext tasks, trunk
save / load), tests in `tests/test_pretraining.py`, launcher `launch_met_pretrain_isambard.sh`.

## 1. Pretext tasks

All tasks work on exactly the tensor the dual model sees, `[B, 2500 nodes, 162 met/static + 16 aux CAMS]`,
scaled by the same scaler class. Channels are `(variable, level, time_delta)`; time deltas are
0, 6 and 12 h **before** the footprint time.

| Task | What is hidden (set to 0 = scaled mean) | Target | Loss | Trivial baseline logged |
|---|---|---|---|---|
| `masked` | every dynamic met channel inside random `block_size`×`block_size` blocks (each block with prob. `mask_ratio`, at least one per sample) | the clean values in those blocks | MSE on masked cells | predict 0 (the mean) |
| `next_delta` | every `time_delta == 0` channel, whole window | the clean `time_delta == 0` fields, i.e. a 6 h forecast from t-6 h / t-12 h | MSE on all cells | persistence of t-6 h |
| `next_delta_tendency` | as `next_delta` | the change `x(t) − x(t-6 h)` | MSE on all cells | predict 0 (= persistence) |
| `delta_forecast` | every dynamic channel whose delta is **not** in `source_deltas`, whole window | the clean fields at `target_delta` (or, with `tendency: true`, their change relative to the nearest source delta) | MSE on all cells | persistence of the source delta nearest in time to the target |
| `delta_forecast` with `remap_sources: true` (shifted window) | **nothing** — the sources are written into the model's input slots (k-th source in time order → k-th slot), so every dynamic channel the trunk sees is a real field | as `delta_forecast` | MSE on all cells | as `delta_forecast` |
| `pseudo_footprint` | nothing (the exact dual input) | **one channel per node**: a kinematic pseudo-footprint computed from the window's winds (§1b) | MSE on all cells | the mean map of the training samples |

`next_delta` is the special case `source_deltas: [6, 12], target_delta: 0` of `delta_forecast`
(the unit tests check the two are identical), so the original configuration is unchanged and
old parameter files run as before. Because time deltas count **backwards**, a target delta larger
than the sources is a *backward* forecast — the direction a footprint describes (where did this
air come from?). Recipes used in `parameter_met_pretrain_transport_2013-17_test2016.json`:

| Run | `source_deltas` → `target_delta` | Reads as | Persistence baseline |
|---|---|---|---|
| `backward6h` | `[0, 6]` → `12` | predict t-12 h from t and t-6 h | t-6 h |
| `backward12h` | `[0]` → `12` | predict t-12 h from t alone (t-6, t-12 zeroed) — 12 h horizon, no local tendency available | t |
| `backward24h` | `[0, 6, 12]` → `24` | predict t-24 h from the full dual input; nothing is hidden | t-12 h |
| `forward12h` | `[12]` → `0` | control: same 12 h horizon, forward in time | t-12 h |
| `shifted6h` | `[6, 12, 18]` → `0`, `remap_sources: true` | `next_delta` with a **full window**: t-6 h in the t slot, t-12 h in the t-6 h slot, t-18 h in the t-12 h slot; nothing hidden (`parameter_met_pretrain_shifted_2013-17_test2016.json`) | t-6 h |
| `future6h` | `[0, 6, 12]` → `-6` | **phase-matched control** of `shifted6h`: predict t+6 h (a *negative* delta = after the footprint time, target-only) from the exact dual input; nothing hidden, no slot shifted (`parameter_met_pretrain_future_2013-17_test2016.json`) | t |

**Target-only time deltas.** A `target_delta` outside the dual layout (24 h above) is requested via
`pretrain.extra_time_deltas: [24]`. The loaders are then run with `variables.time_deltas + extra`,
and the extra channels are appended *after* the model's input columns
(`[met (162) | aux (16) | extra (51)]`), so the trunk still receives exactly the dual model's 178
inputs and `pretrained_trunk` loading stays strict. The input scaler is fit on the dual layout only
(as the fine-tuning run fits it); the extra channels reuse the scaler of the same (variable, level).
Extra deltas may be **negative** (hours *after* the footprint time, e.g. `[-6]` for a t+6 h target):
the loader only computes `fp_time − delta`, and the 3-hourly archive holds those fields too.
Caveat: footprints whose t-24 h met falls before the year's met record (the first day of each year;
the last day for a negative delta) are dropped from *all* channels, so train/val sample counts are a few samples smaller than in a
plain `[6, 12]` load — the pretext validation losses of such runs are therefore not exactly on the
same sample set as the earlier `next_delta` runs (compare downstream fine-tuning instead).

**Shifted window (`remap_sources: true`).** In `next_delta` the 51 `time_delta == 0` channels are
zero for every cell of every sample, so the encoder's first-layer weights on those channels receive
no gradient during pretraining (the input is identically 0) and are transferred untrained — and they
are the channels holding the met at the observation time, the most informative for the footprint.
`remap_sources` removes the zeroed channels: the source deltas may be any *loaded* deltas, one per
input slot of the dual layout (`0, 6, 12`), and the k-th source in time order is written into the
k-th slot. `[6, 12, 18] → 0` (with `extra_time_deltas: [18]`; t-18 h is a stored 3-hourly field like
the others, no interpolation) is therefore the forward 6 h forecast with three real fields in the
three slots; the targets are still the clean t fields (never shown), the persistence baseline is t-6 h,
and `tendency` works as before. The channel layout, count and scaling are unchanged, so the trunk
loads strictly into the dual model — but each slot's *absolute* time is 6 h earlier than at
fine-tuning (slot t holds ~09 UTC fields during pretraining and ~15 UTC fields when fine-tuning, the
GOSAT overpass being fixed), the diurnal confound of §5. The old recipes are untouched (`remap_sources`
defaults to false; with the dual layout itself as sources it reduces to the plain full-input task).

### 1b. The footprint-shaped pretext (`pseudo_footprint`) — recommended when the goal is fp

A footprint is the sensitivity of the observation at the window centre to fluxes in each cell, i.e. the
residence density of air arriving at the release point traced **backward** in time. The closest met-only
analogue is a *kinematic pseudo-footprint* ([`gates/training/pseudo_footprint.py`](../gates/training/pseudo_footprint.py)):
particles are released in the release cell (window index 25, 25) at the footprint time, advected backward
with the boundary-layer winds (each particle follows one model level drawn from `levels`, default 3/9/15 ≈
100/660/1700 m, re-drawn at random every `level_switch_hours` on average as a crude stand-in for
boundary-layer mixing, plus a random walk with diffusivity `diffusion_m2s`), and the hours each particle
spends in each cell are accumulated. Particles that leave the window stop, and the exit side is counted (the
4-side exit distribution the bg head has to infer). The map is scaled with the footprint recipe
`log10(x) + minimum_oom`, zeros left at 0, so the target looks like a scaled footprint: mostly zero, a
plume of a few log-units upwind of the centre.

Why this is the closest configuration to the downstream task:

- **same input** — nothing is hidden, the trunk sees exactly what the dual model sees;
- **same output shape** — one channel per node through the same footprint decoder, so the run can also
  save the decoder (`save_modules: ["encoder", "processor", "fp_decoder"]`) and the fine-tune can
  warm-start the fp head (`pretrained_trunk.modules`);
- **same direction and integration** — backward in time and time-integrated over the arrival window,
  unlike a forecast of a single field at one time.

What it is *not* (principle 1): the only vertical motion is the random level re-draw (no resolved
vertical velocity, same mixing rate day and night), there is no re-entry, wind is 6-hourly (`extra_time_deltas: [18, 24]`) interpolated linearly in time and
bilinearly in space, and the window is treated as a regular lat/lon grid with per-sample spacing. It is a
surrogate label, not an LPDM run, so the fine-tuning still supplies the real physics.

Configuration (`pretrain.pseudo_footprint`, per-run override allowed): `hours` (backward window, needs met
loaded up to that delta), `step_hours`, `levels`, `particles`, `diffusion_m2s`, `level_switch_hours`
(null = particles keep their level), `minimum_oom`, `seed`. Each run saves `pseudo_fp_examples.npz`
(a few validation targets + the baseline map); `scripts/plot_pseudo_footprints.py <run dir>` plots them.
`pretrain.keep_extra_channels: false` drops the extra deltas from the tensors after the targets are made.
`pretrain_settings.json` records the target statistics (fraction of particle-time inside the window,
exit fractions per side, non-zero cells per map). Study file: `parameter_met_pretrain_pseudofp_2013-17_test2016.json`
(24 h with saved fp head, and a 12 h control).

Static channels and aux CAMS are never hidden and never targets. `wind_angle` is hidden with the
rest but is not a target (circular variable, MSE ill-defined). The network is a
`GraphSatelliteDualForecaster` whose footprint decoder is widened to one output per target channel,
so the trunk is built by the same code as in the dual model; only `encoder` + `processor` are saved.
The trunk's `meta` records `pretext` (task, source/target deltas, tendency) and
`input_variable_names` (the 178 input channels in order) for later checks.

## 2. Parameter file

A normal dual parameter file plus a `pretrain` block (see the module docstring of
`train_met_pretrain.py` for every key; example `parameter_met_pretrain_2013-17_test2016.json`).
Rules enforced at start-up:

- `train_load_data.years` must contain every year used by any run and by `scaler_fit_years`;
- pretraining years may **not** overlap `test_load_data.years` (principle 3);
- an existing `trunk_best.pt` is never overwritten (principle 7);
- `delta_forecast` runs need `source_deltas` ⊆ the dual layout (`0` + `variables.time_deltas`) and a
  loaded `target_delta` (dual layout or `pretrain.extra_time_deltas`) that is not a source — checked
  before the data load. With `remap_sources: true` the sources may instead be any loaded deltas but
  must number exactly one per dual-layout slot (three for `[0, 6, 12]`).

`scaler_fit_years` should equal the fine-tuning training years so both stages scale inputs alike.
The test year's **met** is used to report the pretext validation loss and to pick `trunk_best.pt`;
its footprints and backgrounds are never read by the pretraining loop.

## 3. Running

```
sbatch launch_met_pretrain_isambard.sh                                  # all pretrain.runs, one data load
PARAM_FILE=parameter_met_pretrain_smoke.json sbatch --mem=60G --time=00:40:00 launch_met_pretrain_isambard.sh
PARAM_FILE=parameter_met_pretrain_transport_smoke.json sbatch --mem=60G --time=00:40:00 launch_met_pretrain_isambard.sh
PARAM_FILE=parameter_met_pretrain_transport_2013-17_test2016.json sbatch --time=14:00:00 launch_met_pretrain_isambard.sh
PARAM_FILE=parameter_met_pretrain_shifted_smoke.json sbatch --mem=60G --time=00:40:00 launch_met_pretrain_isambard.sh
PARAM_FILE=parameter_met_pretrain_shifted_2013-17_test2016.json sbatch --time=06:00:00 launch_met_pretrain_isambard.sh
PARAM_FILE=parameter_met_pretrain_future_smoke.json sbatch --mem=60G --time=00:40:00 launch_met_pretrain_isambard.sh
PARAM_FILE=parameter_met_pretrain_future_2013-17_test2016.json sbatch --time=06:00:00 launch_met_pretrain_isambard.sh
PARAM_FILE=parameter_met_pretrain_pseudofp_smoke.json sbatch --mem=60G --time=00:40:00 launch_met_pretrain_isambard.sh
PARAM_FILE=parameter_met_pretrain_pseudofp_2013-17_test2016.json sbatch --time=08:00:00 launch_met_pretrain_isambard.sh
```

Parameter files: `parameter_met_pretrain_2013-17_test2016.json` (original `masked` / `next_delta`
study, kept unchanged), `parameter_met_pretrain_transport_2013-17_test2016.json` (the four
`delta_forecast` recipes above, output dir `met_pretrain_sa_test2016_transport`),
`parameter_met_pretrain_shifted_2013-17_test2016.json` (the shifted-window `shifted6h_4yr` run, output
dir `met_pretrain_sa_test2016_shifted`), and the smoke files `parameter_met_pretrain_smoke.json` /
`parameter_met_pretrain_transport_smoke.json` / `parameter_met_pretrain_shifted_smoke.json`.

Outputs: `<save_models_dir>/<pretrain.output_dir>/<run name>/` with `trunk_best.pt`, `trunk_last.pt`,
`pretrain_settings.json` (resolved parameters, channel groups, seed, job id), `pretrain_log.json`
(per-epoch train / val / baseline loss, per-variable val MSE for model and baseline), `updates.txt`.
W&B: one run per pretraining run, `job_type = met_pretrain`.

## 4. Warm-starting a dual run

`train_dual_headlr_model.py` accepts

```json
"pretrained_trunk": {"path": "/abs/path/trunk_best.pt", "modules": ["encoder", "processor"]}
```

Loading is strict: a trunk pretrained with a different input layout or `model_parameters` fails
loudly. `"modules"` may include `"fp_decoder"` for a checkpoint that saved it (a `pseudo_footprint` run
with `save_modules`); asking for a module the checkpoint lacks fails loudly too. With no block the trainer is unchanged. The checkpoint's metadata is written to the run's
`*_updates.txt` and the block is part of `training_settings_*.json` (principle 2). Works with
multi-GPU (rank 0 loads, then the usual weight broadcast). The other dual trainers do not have the
hook yet.

## 5. Comparing pretext tasks

Pretext losses are not comparable across tasks (different targets), so compare *downstream*:
fine-tune each trunk with the identical dual config and put the arms in the same W&B group.
`experiments_ssl_finetune_transport_gpu4_2014-15_test2016.json` holds the four transport arms with
the same overrides and group (`isambard_ssl_pretrain_finetune`) as the original study's scratch and
`nextdelta_4yr` arms (`experiments_ssl_finetune_gpu4_2014-15_test2016_inits.json`), so all of them
sit side by side on W&B; `scripts/compare_ssl_finetune.py` plots them.
`experiments_ssl_finetune_pseudofp_gpu4_2014-15_test2016.json` adds the footprint-shaped arms (24 h trunk
only, 24 h trunk + fp head, 12 h trunk only), and
`experiments_ssl_finetune_shifted_gpu4_2014-15_test2016.json` the shifted-window arm and
`experiments_ssl_finetune_future_gpu4_2014-15_test2016.json` its phase-matched control. Launch (after the
trunks exist):

```
PARAM_FILE=parameter_dual_wsweep_2014-15_test2016.json TRAINER=train_dual_headlr_model \
  EXPERIMENTS_FILE=experiments_ssl_finetune_transport_gpu4_2014-15_test2016.json \
  sbatch --gres=gpu:4 --time=10:00:00 --job-name=ssl_transport_ft launch_dual_refit_isambard.sh
```
