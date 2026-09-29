# How to Predict

> **Status:** this page covers the current prediction script only. It will be expanded.

Run inference with `scripts/predict_GATES_model.py`. Data and model settings are read from the
trained model's `training_outputs/training_settings_<model_name>.json`, so no parameter file is
needed:

```bash
python scripts/predict_GATES_model.py --test_year 2019 --reference_model my_model_20240115_143022
```

`--reference_model` takes the full timestamped folder name (exact match) or the base name
(`my_model`, which picks the most recent run). On SLURM, put this command in place of the
training line in your batch script.

| Argument | Description |
|----------|-------------|
| `--test_year` (required) | Year to predict. |
| `--reference_model` (required) | Trained model folder: full timestamped name, or base name for the latest run. |
| `--month` | One month (`6` or `06`). Default: all 12. |
| `--region` | Override the training region. |
| `--checkpoint` | `best` (default) or an epoch number. |
| `--model_path` | Where to look for trained models (default: from `config.yml`). |
| `--save_path` | Output root (default: the model folder). |
| `--model_save_name` | Output subfolder name under `--save_path` (default: the model's base name). |
| `--size` | Placeholder: only adds `_size{N}` to the output folder name. |
| `--dry_run` | First month only, with `freq=60`. |

## Outputs

One file per month, `predictions_{year}_{MM}.nc`, in a `predictions` folder under the model folder
(or under `{save_path}/{model_save_name}/` if `--save_path` is given). Each file holds
`fp_original`, `fp_transformed`, `fp_pred`, `fp_transformed_pred`, `lat_coords`, `lon_coords` and,
when present, `fp_nan_mask`.

- `--region` adds `_{region}` to the folder name (even if it matches training); `--size` adds `_size{N}`.
- Dry runs write `..._DRYRUN.nc`.
- Each run writes `run_record_<timestamp>.json` with the arguments used.
- Prediction always uses `freq=1` (except for dry runs), loads one month at a time, and does not load flux.

## Not yet supported

Predicting at a different size or on a different domain from training is under development. The
size is always taken from the training settings.
