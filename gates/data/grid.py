"""Order of the nodes in the grid that is handed to the model (``grid_node_order``).

The problem (found 2026-09-29, experiment summary section 22)
-------------------------------------------------------------
The training batches are flattened from ``(sample, lat, lon, variable)``, i.e. latitude-major:
node ``k`` holds the data of cell ``(lat index k // n_lon, lon index k % n_lon)``, so node 1 is
one step in LONGITUDE from node 0.

``gates.data.load_data.get_grid`` lists the positions of the nodes longitude-major: entry ``k`` is
``(lat[k % n_lat], lon[k // n_lat])``, so entry 1 is one step in LATITUDE from entry 0.

The encoder, the mesh and the decoders place node ``k`` at entry ``k`` of that list. For a square
window the model therefore sees the data of cell (lat i, lon j) at the position of cell
(lat j, lon i): the window reflected in its diagonal. Neighbouring cells stay neighbours and the
release point (the centre) stays where it is, so the model trains and tests consistently, but the
mesh edge attributes (distance, latitude / longitude offsets) and the assignment of cells to mesh
nodes are those of the reflected window, in which a step along the data's north-south axis is
given the length of a step in longitude, and the other way round.

The option
----------
``"grid_node_order"`` in the parameter file:

* ``"legacy"`` (default): the list of ``get_grid`` as it is. Every model trained before
  2026-09-29 has this geometry; it is what their saved settings (no key) rebuild.
* ``"latitude_major"``: the same positions in the order of the batches, so every cell's data is
  placed at that cell's own position.

``get_grid`` itself is not changed. Trainers that apply the option record it in
``parameters["grid_node_order_applied"]``; ``require_applied`` (called where the dual model is
built) stops a trainer that does not, instead of letting it ignore the option.

Applied by: ``train_dual_headlr_model.py``, ``predict_dual_model.py``. Not applied (the option
stops them): ``train_dual_model.py``, ``train_dual_refit_model.py``. Not covered by the check:
``train_boundary_model.py``, ``train_met_pretrain.py`` and the footprint-only trainers, which build
their models elsewhere; a trunk pretrained by ``train_met_pretrain.py`` has the legacy geometry.
"""

from .load_data import get_grid

GRID_NODE_ORDERS = ("legacy", "latitude_major")


def resolve_grid_node_order(parameters):
    """Validated ``parameters["grid_node_order"]`` (``"legacy"`` when absent or null)."""
    order = parameters.get("grid_node_order", None)
    if order is None:
        return "legacy"
    if order not in GRID_NODE_ORDERS:
        raise ValueError(f"grid_node_order must be one of {list(GRID_NODE_ORDERS)}, got {order!r}")
    return order


def reorder_latitude_major(entries, n_lat, n_lon):
    """Entries of a ``get_grid`` list (longitude-major) in latitude-major order.

    ``get_grid`` puts cell (lat index j, lon index i) at position ``i * n_lat + j``; the result
    has it at position ``j * n_lon + i``, the position of that cell's data in a flattened batch.
    """
    if len(entries) != n_lat * n_lon:
        raise ValueError(f"grid of {len(entries)} nodes does not match a {n_lat} x {n_lon} window")
    return [entries[i * n_lat + j] for j in range(n_lat) for i in range(n_lon)]


def get_model_grid(fp_xr, parameters):
    """The grid of ``get_grid`` in the node order requested by ``parameters``.

    Args:
        fp_xr (xr.Dataset): Footprint data with ``lat_coords`` / ``lon_coords`` and the
            coordinates time, lat, lon (as for ``get_grid``).
        parameters (dict): Reads ``grid_reference_fp`` and ``grid_node_order``; writes
            ``grid_node_order_applied`` (saved with the training settings).

    Returns:
        tuple: ``(latlons, idx_latlons)`` as ``get_grid`` returns them; with
        ``"latitude_major"`` entry ``k`` belongs to the cell whose data is node ``k`` of a batch.
    """
    order = resolve_grid_node_order(parameters)
    latlons, idx_latlons = get_grid(fp_xr, parameters.get("grid_reference_fp"))
    if order == "latitude_major":
        n_lat, n_lon = int(fp_xr.lat.size), int(fp_xr.lon.size)
        latlons = reorder_latitude_major(latlons, n_lat, n_lon)
        idx_latlons = reorder_latitude_major(idx_latlons, n_lat, n_lon)
        print(f"Grid node order: latitude-major ({n_lat} x {n_lon} cells): every cell's data is placed at "
              "its own position (the default, 'legacy', places it at the position of the cell reflected "
              "in the diagonal of the window)")
    parameters["grid_node_order_applied"] = order
    return latlons, idx_latlons


def require_applied(parameters):
    """Fail loudly when a non-default node order is requested but the trainer never applied it."""
    order = resolve_grid_node_order(parameters)
    if order != "legacy" and parameters.get("grid_node_order_applied") != order:
        raise ValueError(f"grid_node_order is {order!r} but the grid of this run was not built with it: this "
                         "trainer does not apply the option (train_dual_headlr_model and "
                         "predict_dual_model do)")
