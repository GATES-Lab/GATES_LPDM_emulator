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
"""

import copy


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
