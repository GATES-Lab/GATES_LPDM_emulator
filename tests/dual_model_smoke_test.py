"""Self-contained smoke test for GraphSatelliteDualForecaster.

Builds the dual-head model on a small synthetic grid and runs forward + backward through
*both* decoder heads (footprint + background), for both decoder types and num_classes
settings. This validates the new model wiring / dict output / joint-loss path without needing
the real footprint or CAMS data (which are not available in every environment).
"""

import os
import sys

# repo root (parent of tests/) so the local `model` and `gates` packages import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, "/user/work/yl18410/new_graphnet")
sys.path.insert(2, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator")

import numpy as np
import torch

from model.forecast import GraphSatelliteDualForecaster
import gates.evaluation.loss_functions as gates_losses


def make_grid(side):
    """Regular lat/lon grid of `side` x `side` nodes -> list of (lat, lon)."""
    lats = np.linspace(-5.0, 5.0, side)
    lons = np.linspace(-65.0, -55.0, side)
    mg = np.meshgrid(lats, lons)
    return [(mg[0][i, j], mg[1][i, j]) for i in range(side) for j in range(side)]


def run_case(decoder_type, num_classes, side=8, batch=3, feature_dim=5, aux_dim=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lat_lons = make_grid(side)
    n_nodes = side * side

    model = GraphSatelliteDualForecaster(
        lat_lons,
        whole_world=False,
        resolution=4,
        feature_dim=feature_dim,
        aux_dim=aux_dim,
        num_classes=num_classes,
        decoder_type=decoder_type,
        input_height=side,
        input_width=side,
        fp_output_dim=1,
        node_dim=16,
        edge_dim=16,
        num_blocks=2,
        hidden_dim_processor_node=16,
        hidden_dim_processor_edge=16,
        hidden_dim_decoder=16,
        hidden_layers_decoder=1,
        output_dim=8,
    ).to(device)

    features = torch.randn(batch, n_nodes, feature_dim + aux_dim, device=device)
    fp_target = torch.randn(batch, n_nodes, 1, device=device)
    bg_target = torch.randn(batch, num_classes, device=device)

    out = model(features)
    assert isinstance(out, dict) and set(out) == {"footprint", "background"}, out.keys()
    fp_pred, bg_pred = out["footprint"], out["background"]

    assert fp_pred.shape == (batch, n_nodes, 1), f"fp shape {fp_pred.shape}"
    assert bg_pred.shape == (batch, num_classes), f"bg shape {bg_pred.shape}"

    fp_criterion = gates_losses.MSELoss()
    bg_criterion = torch.nn.MSELoss()
    fp_loss = fp_criterion(fp_pred, fp_target)
    bg_loss = bg_criterion(bg_pred, bg_target)
    loss = fp_loss + 1.0 * bg_loss

    model.zero_grad()
    loss.backward()

    # confirm both heads received gradients
    fp_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fp_decoder.parameters())
    bg_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.bg_decoder.parameters())
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert fp_grad, "footprint head got no gradient"
    assert bg_grad, "background head got no gradient"
    assert enc_grad, "shared encoder got no gradient"

    print(f"  [OK] decoder={decoder_type} num_classes={num_classes} "
          f"fp={tuple(fp_pred.shape)} bg={tuple(bg_pred.shape)} "
          f"loss={loss.item():.4f} (fp={fp_loss.item():.4f}, bg={bg_loss.item():.4f})")


if __name__ == "__main__":
    print("device:", "cuda" if torch.cuda.is_available() else "cpu")
    for decoder_type in ["conv", "dense"]:
        for num_classes in [1, 4]:
            run_case(decoder_type, num_classes)
    print("ALL DUAL MODEL SMOKE TESTS PASSED")
