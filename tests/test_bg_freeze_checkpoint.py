"""Tests for per-head checkpointing and bg-head freezing in the dual model.

Covers:
  - HeadCheckpoint: saves on improvement, keeps the best epoch's weights through stagnant
    epochs, counts stagnation, and respects ``delta``.
  - setup_dual_model: the optional ``bg_head`` block builds a bg-decoder-only optimizer
    parameter group (weight decay / LR scale) and wires the freeze settings into the
    model context, while the default config keeps a single parameter group.
  - freeze_bg_head + the frozen branch of train_one_epoch: the bg decoder stops updating
    (and stays in eval mode) while the shared encoder and fp head keep training.

Run via SLURM (never on the login node): ``sbatch launch_test_bg_freeze.sh``
"""

import os
import sys

# repo root (parent of tests/) so the local `model` and `gates` packages import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, "/user/work/yl18410/new_graphnet")
sys.path.insert(2, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")

import numpy as np
import torch

import gates.evaluation.loss_functions as gates_losses
import gates.training.training_dual as gates_training_dual
from gates.training.training_dataclasses import DualModelContext, BoundaryTrainingContext, PathContext
from gates.training.training_helperfuns import HeadCheckpoint
from model.forecast import GraphSatelliteDualForecaster
from train_dual_model import train_one_epoch


# ---------------------------------------------------------------
# HeadCheckpoint
# ---------------------------------------------------------------

def test_head_checkpoint_saves_best_and_counts_stagnation(tmp_path):
    model = torch.nn.Linear(3, 2)
    path = tmp_path / "best_fp.pt"
    ckpt = HeadCheckpoint("fp", path, verbose=False)

    assert ckpt.step(1.0, model, epoch=0)  # first epoch always improves from inf
    best_weight = model.weight.detach().clone()

    with torch.no_grad():
        model.weight += 1.0  # live model moves on...
    assert not ckpt.step(1.5, model, epoch=1)  # ...but these epochs are worse: no save
    assert not ckpt.step(1.2, model, epoch=2)
    assert ckpt.counter == 2
    assert ckpt.best_epoch == 0
    assert torch.allclose(torch.load(path)["weight"], best_weight)

    assert ckpt.step(0.5, model, epoch=3)  # improvement: save + reset counter
    assert ckpt.counter == 0
    assert ckpt.best_epoch == 3
    assert torch.allclose(torch.load(path)["weight"], model.weight)


def test_head_checkpoint_min_delta(tmp_path):
    model = torch.nn.Linear(2, 1)
    ckpt = HeadCheckpoint("bg", tmp_path / "best_bg.pt", delta=0.1, verbose=False)

    assert ckpt.step(1.0, model, epoch=0)
    assert not ckpt.step(0.95, model, epoch=1)  # improves, but by less than delta
    assert ckpt.counter == 1
    assert ckpt.step(0.85, model, epoch=2)  # improves by more than delta
    assert ckpt.best_loss == 0.85


# ---------------------------------------------------------------
# Small real dual model + context helpers
# ---------------------------------------------------------------

SIDE = 8
FEATURE_DIM = 5
AUX_DIM = 4
NUM_CLASSES = 1


def make_grid(side=SIDE):
    """Regular lat/lon grid of `side` x `side` nodes -> list of (lat, lon)."""
    lats = np.linspace(-5.0, 5.0, side)
    lons = np.linspace(-65.0, -55.0, side)
    mg = np.meshgrid(lats, lons)
    return [(mg[0][i, j], mg[1][i, j]) for i in range(side) for j in range(side)]


SMALL_MODEL_PARAMS = dict(
    num_classes=NUM_CLASSES, fp_output_dim=1, node_dim=16, edge_dim=16, num_blocks=2,
    hidden_dim_processor_node=16, hidden_dim_processor_edge=16, hidden_dim_decoder=16,
    hidden_layers_decoder=1, output_dim=8, resolution=4,
)


def small_dual_model():
    return GraphSatelliteDualForecaster(
        make_grid(), whole_world=False, feature_dim=FEATURE_DIM, aux_dim=AUX_DIM,
        decoder_type="conv", input_height=SIDE, input_width=SIDE, **SMALL_MODEL_PARAMS,
    )


def make_ctx(model, bg_loss_weight=0.2):
    """Minimal DualModelContext for exercising train_one_epoch on CPU."""
    return DualModelContext(
        model_name="freeze_test", use_wandb=False, device=torch.device("cpu"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        fp_criterion=gates_losses.MSELoss(), fp_criterion_test=gates_losses.MSELoss(),
        bg_criterion=torch.nn.MSELoss(), bg_criterion_test=torch.nn.MSELoss(),
        bg_loss_weight=bg_loss_weight, lr=1e-3, early_stopping=None,
        epochs_num=1, epochs_visualise=1, epochs_save=1, epochs_patience=1,
    )


def make_loader(n_batches=2, batch=3):
    n_nodes = SIDE * SIDE
    return [
        (torch.randn(batch, n_nodes, FEATURE_DIM + AUX_DIM),
         torch.randn(batch, n_nodes),
         torch.randn(batch, NUM_CLASSES))
        for _ in range(n_batches)
    ]


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _any_changed(module, before):
    return any(not torch.equal(p.detach(), b) for p, b in zip(module.parameters(), before))


# ---------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------

def test_train_one_epoch_unfrozen_updates_all_parts():
    torch.manual_seed(0)
    model = small_dual_model()
    model_ctx = make_ctx(model)
    enc0, fp0, bg0 = _snapshot(model.encoder), _snapshot(model.fp_decoder), _snapshot(model.bg_decoder)

    total, fp_loss, bg_loss = train_one_epoch(model, make_loader(), model_ctx, epoch=0)

    assert np.isfinite([total, fp_loss, bg_loss]).all()
    assert _any_changed(model.encoder, enc0), "shared encoder did not update"
    assert _any_changed(model.fp_decoder, fp0), "fp head did not update"
    assert _any_changed(model.bg_decoder, bg0), "bg head did not update"


def test_train_one_epoch_frozen_bg_head_stops_updating():
    torch.manual_seed(0)
    model = small_dual_model()
    model_ctx = make_ctx(model)  # optimizer built before the freeze, as in a real run

    gates_training_dual.freeze_bg_head(model)
    model_ctx.bg_frozen = True
    assert all(not p.requires_grad for p in model.bg_decoder.parameters())

    enc0, fp0, bg0 = _snapshot(model.encoder), _snapshot(model.fp_decoder), _snapshot(model.bg_decoder)
    total, fp_loss, bg_loss = train_one_epoch(model, make_loader(), model_ctx, epoch=0)

    assert np.isfinite([total, fp_loss, bg_loss]).all()
    assert bg_loss > 0, "frozen bg loss should still be computed for logging"
    assert not _any_changed(model.bg_decoder, bg0), "frozen bg head must not update"
    assert _any_changed(model.encoder, enc0), "shared encoder should keep training"
    assert _any_changed(model.fp_decoder, fp0), "fp head should keep training"
    # train_one_epoch keeps the frozen head in eval mode while the rest trains
    assert not model.bg_decoder.training
    assert model.encoder.training


# ---------------------------------------------------------------
# setup_dual_model: bg_head config block
# ---------------------------------------------------------------

def base_parameters():
    return {
        "model_name": "freeze_setup_test",
        "use_wandb": False,
        "verbose": False,
        "learning_rate": 1e-4,
        "model_parameters": {"decoder": "conv", **SMALL_MODEL_PARAMS},
        "dataloader": {"nans_to_zeros": False},
        "epochs": {"training": 1, "patience": 5, "model_save": 1},
        "loss_functions": {
            "fp_criterion": "gates_losses.MSELoss",
            "bg_criterion": "torch.nn.MSELoss",
            "bg_loss_weight": 0.2,
        },
    }


def setup_ctx(tmp_path, parameters):
    paths_ctx = PathContext(model_save_dir=str(tmp_path), model_name="freeze_setup_test",
                            model_path=tmp_path)
    training_ctx = BoundaryTrainingContext(
        parameters, "cpu", False, [], [], make_grid(), ["fp_transformed"], {},
        FEATURE_DIM, SIDE, AUX_DIM,
    )
    return gates_training_dual.setup_dual_model(parameters, training_ctx, paths_ctx)


def test_setup_dual_model_default_single_param_group(tmp_path):
    model, model_ctx = setup_ctx(tmp_path, base_parameters())
    assert len(model_ctx.optimizer.param_groups) == 1
    assert model_ctx.bg_freeze_patience is None
    assert model_ctx.bg_frozen is False


def test_setup_dual_model_bg_head_block(tmp_path):
    parameters = base_parameters()
    parameters["bg_head"] = {"freeze_patience": 10, "freeze_min_delta": 0.01,
                             "weight_decay": 0.05, "lr_scale": 0.5}
    model, model_ctx = setup_ctx(tmp_path, parameters)

    groups = model_ctx.optimizer.param_groups
    assert len(groups) == 2
    shared_group, bg_group = groups
    assert bg_group["weight_decay"] == 0.05
    assert bg_group["lr"] == 1e-4 * 0.5
    assert shared_group["lr"] == 1e-4

    # the bg group is exactly the bg decoder's params; together the groups cover the model
    bg_ids = {id(p) for p in model.bg_decoder.parameters()}
    assert {id(p) for p in bg_group["params"]} == bg_ids
    assert len(shared_group["params"]) + len(bg_group["params"]) == len(list(model.parameters()))

    assert model_ctx.bg_freeze_patience == 10
    assert model_ctx.bg_freeze_min_delta == 0.01
