"""Helpers to run several experiments from one base parameter file + a set of overrides.

An *experiments* file lists named experiments, each with an ``overrides`` dict that is deep-merged
onto the base parameters. Overrides may be nested dicts mirroring the parameter structure, or use
dotted keys for brevity, e.g.::

    {
      "experiments": [
        {"name": "big_nodes",   "overrides": {"model_parameters": {"node_dim": 128}}},
        {"name": "bg_weight_2",  "overrides": {"loss_functions.bg_loss_weight": 2.0}}
      ]
    }

Both forms above are equivalent in style; pick whichever is clearer. The experiment ``name`` is
appended to ``model_name`` so each run is identifiable (the usual timestamp is still added on top).

A second, more compact way to generate experiments is a ``__sweep__`` section embedded directly in
the base parameter file (this is the style used by ``train_GATES_sweep.py``). It is expanded here
into the same experiment-entry form so both drivers share one code path — see :func:`expand_sweep`
and :func:`sweep_to_experiments`.
"""

import copy
from itertools import product


# Overriding any of these per experiment/sweep combination changes what data should be loaded, so a
# shared data bundle (loaded once and reused across runs) would no longer be valid for that run.
# Shared-data drivers reject sweeps/experiments that touch these keys.
DATA_LOADING_KEYS = {
    "train_load_data", "test_load_data", "variables", "background_setup",
    "data_dirs", "load_into_memory",
}


def expand_dotted_keys(d):
    """Expand any dotted top-level keys (``"a.b.c": v``) into nested dicts.

    Non-dotted keys are kept as-is. Nested dict values are expanded recursively so a mix of
    dotted and nested forms works. Returns a new dict; the input is not modified.
    """
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = expand_dotted_keys(value)
        if isinstance(key, str) and "." in key:
            parts = key.split(".")
            node = out
            for part in parts[:-1]:
                existing = node.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    node[part] = existing
                node = existing
            # merge in case the leaf already partially exists
            if isinstance(value, dict) and isinstance(node.get(parts[-1]), dict):
                node[parts[-1]] = deep_update(node[parts[-1]], value)
            else:
                node[parts[-1]] = value
        else:
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = deep_update(out[key], value)
            else:
                out[key] = value
    return out


def deep_update(base, overrides):
    """Recursively merge ``overrides`` into ``base`` and return the result.

    Dict values are merged key-by-key; any non-dict value (including lists) replaces the base
    value outright. ``base`` is deep-copied, so neither input is mutated.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def build_experiment_params(base_params, experiment):
    """Build the parameter dict for a single experiment.

    Args:
        base_params (dict): The base parameters (not modified).
        experiment (dict): An experiment entry with optional ``name`` and ``overrides`` keys.

    Returns:
        (name, params): the experiment name and the merged parameter dict, with the experiment
        name suffixed onto ``model_name``.
    """
    name = experiment.get("name")
    overrides = expand_dotted_keys(experiment.get("overrides", {}))
    params = deep_update(base_params, overrides)
    if name:
        base_model_name = params.get("model_name", "model")
        params["model_name"] = f"{base_model_name}_{name}"
    return name, params


def load_experiments(experiments_obj):
    """Normalise an experiments definition into a list of experiment dicts.

    Accepts either a dict with an ``"experiments"`` list, or a bare list of experiment entries.
    """
    if isinstance(experiments_obj, dict):
        experiments = experiments_obj.get("experiments", [])
    elif isinstance(experiments_obj, list):
        experiments = experiments_obj
    else:
        raise ValueError("experiments file must be a JSON list or an object with an 'experiments' list")
    if not experiments:
        raise ValueError("no experiments found in the experiments file")
    return experiments


# --- Sweep support (the "__sweep__" section style used by train_GATES_sweep.py) ----------------

def format_value(v):
    """Format a single value compactly for use in file/run/model names."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v:.2g}"
    if isinstance(v, (list, dict)):
        return "complex"
    return str(v)


def make_suffix(combo):
    """Build a readable suffix string from a ``{dot_key: value}`` combo dict.

    Uses the last segment of each key path unless two keys share the same last segment, in which
    case the full dotted path (with dots turned into dashes) is used to keep the suffix unambiguous.
    """
    short_keys = [kp.split(".")[-1] for kp in combo]
    use_full = len(set(short_keys)) < len(short_keys)
    parts = []
    for key_path, value in combo.items():
        label = key_path.replace(".", "-") if use_full else key_path.split(".")[-1]
        parts.append(f"{label}-{format_value(value)}")
    return "_".join(parts)


def expand_sweep(sweep_spec):
    """Expand a ``__sweep__`` specification into a list of ``{dot_key: value}`` combo dicts.

    Two forms are accepted (mirroring ``train_GATES_sweep.py``):

    * **Cartesian product** — a dict of ``{dot_key: [values]}``. Every combination of the listed
      values is produced.
    * **Explicit combinations** — a list of ``{dot_key: value}`` dicts, used verbatim.

    Dot-notation addresses nested keys at any depth (e.g. ``"model_parameters.num_blocks"``).
    """
    if isinstance(sweep_spec, list):
        return [dict(combo) for combo in sweep_spec]
    if isinstance(sweep_spec, dict):
        keys = list(sweep_spec.keys())
        values = [sweep_spec[k] for k in keys]
        return [dict(zip(keys, vals)) for vals in product(*values)]
    raise ValueError("'__sweep__' must be a dict of {key: [values]} or a list of {key: value} dicts")


def sweep_to_experiments(sweep_spec):
    """Turn a ``__sweep__`` spec into a list of experiment entries for :func:`build_experiment_params`.

    Each combination becomes ``{"name": <id>_<suffix>, "overrides": combo, "sweep_id": i,
    "sweep_combination": combo}``. The overrides use the combo's dotted keys directly, which
    :func:`build_experiment_params` expands and deep-merges onto the base parameters. This lets the
    ``__sweep__`` style and the explicit ``experiments`` style share one execution path.
    """
    combos = expand_sweep(sweep_spec)
    experiments = []
    for i, combo in enumerate(combos):
        sweep_id = i + 1
        experiments.append({
            "name": f"sweep_{sweep_id:03d}_{make_suffix(combo)}",
            "overrides": dict(combo),
            "sweep_id": sweep_id,
            "sweep_combination": dict(combo),
        })
    return experiments


def data_loading_clashes(overrides):
    """Return the set of top-level :data:`DATA_LOADING_KEYS` touched by an ``overrides`` dict.

    ``overrides`` may use dotted or nested keys; both are expanded before checking. An empty set
    means the overrides are safe to run against a shared (loaded-once) data bundle.
    """
    touched = set(expand_dotted_keys(overrides).keys())
    return touched & DATA_LOADING_KEYS
