# How to Train on Multiple Regions

> **Status:** this page covers the current multi-region trainer only. It will be expanded.

To train one model on footprints from several regions, use
`scripts/train_GATES_model_multiregion.py`. It takes the same CLI as the standard trainer:

```bash
python scripts/train_GATES_model_multiregion.py parameter_file.json
```

It does not work with sweeps yet.

## Parameter file

Replace the top-level `train_load_data` / `test_load_data` with:

- **`regions`:** a dict keyed by `"<int>"` or `"<int>-<label>"`. The integer sets the order. Each
  entry has its own `train_load_data` and `test_load_data`.
- **`shared_load_parameters`:** the loading keys common to all regions (e.g. `size`, `freq`,
  `crop_met`). Per-region keys override them, and each `test_load_data` is merged on top of its
  `train_load_data`.

```json
"shared_load_parameters": {"size": 50, "freq": 3, "crop_met": false},
"regions": {
    "0-SAHARA": {
        "train_load_data": {"region": "SAHARA", "years": ["2014", "2015"]},
        "test_load_data":  {"years": ["2016"], "freq": 100}
    },
    "1-BRAZIL": {
        "train_load_data": {"region": "BRAZIL", "years": ["2014", "2015"]},
        "test_load_data":  {"years": ["2016"], "freq": 100}
    }
}
```

All other sections are shared across regions. Every region must use the same `size` and the same
`variables`; the trainer raises an error otherwise. The multi-region trainer ignores
`review_fixes.use_tensor_loader` and always uses the xbatcher loader, so set worker options in
`dataloader_params`.

Templates: [`NEW_parameter_template_multiregion.json`](../parameter_files/NEW_parameter_template_multiregion.json)
and the smoke test [`NEW_parameter_template_terminal_multiregion.json`](../parameter_files/NEW_parameter_template_terminal_multiregion.json).

### Flux

The top-level `flux` block (see [HOW_TO_PARAMETER_FILE.md](HOW_TO_PARAMETER_FILE.md#flux)) is the
default for every region. A region can override it with its own `flux` block, placed next to
`train_load_data` (not inside it):

```json
"flux": {"get_flux": true, "flux_time": "2016",
         "convert_units": true, "convert_units_args": {"unit_multiplier": 1e9}},
"regions": {
    "0-SAHARA": {"train_load_data": {...}, "test_load_data": {...}},
    "1-BRAZIL": {"train_load_data": {...}, "test_load_data": {...},
                 "flux": {"flux_time": "2016-01"}},
    "2-CHINA":  {"train_load_data": {...}, "test_load_data": {...},
                 "flux": {"get_flux": false}}
}
```

- The merge is one level deep: a region's keys replace the top-level ones. A region that sets
  `convert_units_args` replaces the whole sub-dict.
- `"get_flux": false` turns flux off for that region without a warning. It gets no flux metrics.
- Each region's resolved settings are saved as `resolved_region_flux` in the training settings and
  the W&B config.
- `"flux_time": "2016"` is recommended (it is the templates' default). It gives every region and
  test month the same fixed flux, so the flux metrics are comparable.

Flux is loaded for the **test data only**, since only the flux metrics use it. The training data
loads flux only if the loss is flux-weighted (`weight_label: "flux"` in `criterion_params` or
`criterion_test_params`). In that case every region must have `get_flux` on, and every training
footprint must have flux; otherwise training stops with an error.

A missing flux file or month only warns. That region's footprints are kept with NaN flux, and the
region gets no flux metrics. At the time of writing this applies to CHINA outside 2016 and to
INDIA (no flux files).

## During and after training

- After loading, the updates file (and stdout) lists each region's test flux, so you can see
  which regions will get flux metrics:
  ```
  Flux per region:
    0-SAHARA: test flux yes (flux_time=2016)
    2-CHINA: test flux NO (all NaN) (flux_time=None)
    3-INDIA: test flux off
  ```
- The training data from all regions is combined, and the scalers are fitted once on it.
- The model is validated on each region every epoch. Early stopping uses the aggregate test loss.
- W&B keys are grouped by region: `metrics_original/<name>/...`, `metrics_transformed/<name>/...`
  and `metrics_fluxes_static/<name>/<mode>/...`, where `<name>` is a region (`0-SAHARA`) or
  `aggregate`.
- Metrics against the loaded flux are per region only: `metrics_fluxes/<region>/...`, logged
  only for regions whose test set has flux. There is no aggregate. They are also added to each
  region's line in the updates file.
- The training plots use the first test region. They are saved every `epochs.visualize` epochs and
  uploaded to W&B every 3 × `visualize` epochs.
- At the end, one test prediction file is written per region: `sample_predictions_test_<idx>-<region>.nc`.

## To Dos:
- Adapt to seeds
- Add per-domain test figures
- Introduce tensor dataloader