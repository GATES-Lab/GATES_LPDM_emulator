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

## During and after training

- The training data from all regions is combined, and the scalers are fitted once on it.
- The model is validated on each region every epoch. Early stopping uses the aggregate test loss.
- W&B keys are grouped by region: `metrics_original/<name>/...`, `metrics_transformed/<name>/...`
  and `metrics_fluxes_static/<name>/<mode>/...`, where `<name>` is a region (`0-SAHARA`) or
  `aggregate`.
- The training plots use the first test region. They are saved every `epochs.visualize` epochs and
  uploaded to W&B every 3 × `visualize` epochs.
- At the end, one test prediction file is written per region: `sample_predictions_test_<idx>-<region>.nc`.

## To Dos:
- Adapt to seeds
- Add per-domain test figures
- Introduce tensor dataloader