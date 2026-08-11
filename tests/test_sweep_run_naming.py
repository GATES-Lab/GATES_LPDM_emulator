"""Tests for the automatic W&B run naming of sweep trials (run_dual_sweep_wandb.py).

Each trial run is renamed right after ``wandb.init()`` to
``{SLURM_JOB_ID}_{SLURM_JOB_NAME}_{param}={value}_...`` — the same prefix
train_and_save_model gives non-sweep runs, plus one token per sweep-chosen
parameter (common parameters abbreviated, floats to 3 significant digits).

Run via SLURM (never on the login node): ``sbatch launch_test_sweep_naming.sh``
"""

import os
import sys

# repo root (parent of tests/) so the local `gates` package and the sweep driver import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, "/user/work/yl18410/new_graphnet")
sys.path.insert(2, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")

from run_dual_sweep_wandb import trial_run_name


def test_name_has_slurm_prefix_and_all_sweep_params(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_NAME", "dual_wandb_sweep")
    # The full parameter set of sweep_dual_wandb.yaml, with a raw log-uniform lr draw.
    name = trial_run_name({
        "learning_rate": 1.6318772126672317e-05,
        "loss_functions.bg_loss_weight": 0.05,
        "model_parameters.dropout": 0.0,
        "model_parameters.num_blocks": 4,
        "model_parameters.node_dim": 128,
        "model_parameters.hidden_dim_decoder": 64,
        "epochs.training": 150,
    })
    assert name == ("12345_dual_wandb_sweep_"
                    "ep=150_lr=1.63e-05_bgw=0.05_drop=0_dec=64_node=128_blocks=4")


def test_unabbreviated_params_fall_back_to_last_dotted_segment(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NAME", "j")
    assert trial_run_name({"bg_head.freeze_patience": 25}) == "1_j_freeze_patience=25"


def test_deterministic_order_regardless_of_dict_order(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_JOB_NAME", "j")
    a = trial_run_name({"learning_rate": 1e-4, "model_parameters.dropout": 0.1})
    b = trial_run_name({"model_parameters.dropout": 0.1, "learning_rate": 1e-4})
    assert a == b == "1_j_lr=0.0001_drop=0.1"


def test_without_slurm_env_uses_local_run_prefix(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    assert trial_run_name({}) == "local_run"
