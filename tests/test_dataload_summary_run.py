"""Unit test for the separate W&B data-loading summary run (no real data needed).

Verifies the shared-data logging changes in train_dual_model.py by stubbing the expensive
loader, so it runs in seconds on any node:

  1. ``load_dual_data(..., return_summary=True)`` returns ``(bundle, summary)`` with the
     expected timing/sample-count keys (per-month times plumbed out of
     ``load_GATES_data_with_bg`` via ``loading_times_out``);
  2. the default call still returns just the bundle (backward compatibility for
     train_and_save_model and any other caller);
  3. ``log_data_loading_summary_run`` creates its own W&B run and leaves NO active run
     behind (checked in wandb offline mode — no network or login needed);
  4. it refuses to hijack an already-active run (the sweep-reuse logic in
     train_and_save_model depends on this);
  5. it skips cleanly when wandb entity/project are missing.

Run with: python tests/test_dataload_summary_run.py
"""

import os
import sys
import tempfile

# repo root (parent of tests/) so the local `gates` package and train_dual_model import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, "/user/work/yl18410/new_graphnet")
sys.path.insert(2, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")

# Offline mode: no network, no login; keep the offline run files out of the repo wandb dir.
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = tempfile.mkdtemp(prefix="wandb_dataload_test_")
os.environ["WANDB_SILENT"] = "true"

import wandb

import train_dual_model as tdm
from gates.training.training_dataclasses import PathContext


class _FakeXr:
    """Stands in for the loaded xr.Dataset — load_dual_data only touches .time on it."""

    def __init__(self, n):
        self.time = list(range(n))


def _fake_load_gates_data_with_bg(data_parameters, input_variables=None, datapath_args=None,
                                  verbose=True, load_into_memory=True, detrend=True,
                                  use_aux_bc=True, aux_indeces=None, loading_times_out=None):
    if loading_times_out is not None:
        loading_times_out["2014-1"] = 1.25
        loading_times_out["2014-2"] = 2.5
    return _FakeXr(10), "inputs", "bgs", "aux"


def main():
    # --- stub the expensive/environment-dependent pieces ---
    tdm.gates_training_background.load_GATES_data_with_bg = _fake_load_gates_data_with_bg
    tdm.gates_training.make_cluster = lambda: (None, None)
    PathContext.resolve_datapath_args = lambda self, p: {}

    params = {
        "train_load_data": {"years": [2014], "months": [1, 2]},
        "test_load_data": {"years": [2016]},
        "variables": {"dummy": True},
        "use_wandb": True,
        "wandb": {"entity": "test-entity", "project": "test-project",
                  "group": "test-group", "tags": ["smoke"]},
    }

    # 1. return_summary=True -> (bundle, summary) with the expected contents
    bundle, summary = tdm.load_dual_data(params, verbose=False, return_summary=True)
    assert isinstance(bundle, tdm.DualDataBundle), type(bundle)
    expected_keys = {"train_load_mins", "test_load_mins", "total_load_mins",
                     "n_train_samples", "n_test_samples", "train_month_mins", "test_month_mins"}
    assert expected_keys == set(summary.keys()), summary.keys()
    assert summary["n_train_samples"] == 10 and summary["n_test_samples"] == 10
    assert summary["train_month_mins"] == {"2014-1": 1.25, "2014-2": 2.5}
    assert summary["total_load_mins"] == summary["train_load_mins"] + summary["test_load_mins"]
    print("PASS 1: load_dual_data(return_summary=True) returns (bundle, summary)")

    # 2. default call unchanged (backward compatibility)
    bundle_only = tdm.load_dual_data(params, verbose=False)
    assert isinstance(bundle_only, tdm.DualDataBundle), type(bundle_only)
    print("PASS 2: load_dual_data() default return unchanged")

    # 3. the summary run is created and finished — no active run left behind
    assert wandb.run is None
    tdm.log_data_loading_summary_run(params, summary)
    assert wandb.run is None, "log_data_loading_summary_run left an active W&B run behind!"
    print("PASS 3: summary run created and finished; no active run leaks")

    # 4. never hijacks an already-active run
    existing = wandb.init(project="test-project")
    tdm.log_data_loading_summary_run(params, summary)
    assert wandb.run is existing, "log_data_loading_summary_run replaced the active run!"
    wandb.finish()
    print("PASS 4: active run untouched")

    # 5. missing entity/project -> clean skip
    tdm.log_data_loading_summary_run({"wandb": {}}, summary)
    assert wandb.run is None
    print("PASS 5: clean skip without wandb entity/project")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
