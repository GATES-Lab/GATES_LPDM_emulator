"""Build (or complete) the on-disk month cache of a dual-model parameter file, without training.

Training jobs with ``"data_cache": {"enabled": true}`` build the months they miss themselves; this
script does the same ahead of time, and can split the work over several jobs::

    python prepare_dual_data_cache.py parameter_dual_large.json
    python prepare_dual_data_cache.py parameter_dual_large.json --shard 0/4      # job 1 of 4
    python prepare_dual_data_cache.py parameter_dual_large.json --split test --rebuild

Months that are already cached (and whose footprint files did not change) are skipped, so the
script can be re-run to complete a cache. See How_Tos/HOW_TO_LARGE_DATASETS.md.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import gates
import gates.training.lean_dual_data as gates_lean
from gates.training.training_dataclasses import PathContext
from gates.training.training_helperfuns import load_parameter_file
from train_dual_model import resolve_background_params


def main():
    parser = argparse.ArgumentParser(description="Build the month cache of a dual-model parameter file.")
    parser.add_argument("file_name", help="Parameter file name")
    parser.add_argument("--file_path", default=None,
                        help="Parameter file path. Defaults to config.yml's parameter_files_dir.")
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="build only the months at position I modulo N of the month list")
    parser.add_argument("--rebuild", action="store_true", help="reload months that are already cached")
    args = parser.parse_args()

    cfg = gates.config.get_config()
    file_path = args.file_path if args.file_path is not None else cfg.parameter_files_dir
    parameters = load_parameter_file(Path(file_path) / args.file_name)
    if parameters is None:
        print("Error loading parameters. Exiting.")
        sys.exit(1)
    parameters.setdefault("data_cache", {})["enabled"] = True
    settings = gates_lean.data_cache_settings(parameters)

    shard = None
    if args.shard is not None:
        i, n = (int(v) for v in args.shard.split("/"))
        if not 0 <= i < n:
            parser.error(f"--shard {args.shard}: need 0 <= I < N")
        shard = (i, n)

    train_params = copy.deepcopy(parameters["train_load_data"])
    test_params = copy.deepcopy(parameters["train_load_data"])
    test_params.update(parameters["test_load_data"])
    datapath_args = PathContext(model_save_dir=".", model_name="_dataload",
                                model_path=Path(".")).resolve_datapath_args(parameters)
    background_params = resolve_background_params(parameters)
    verbose = parameters.get("verbose", True)

    print(f"Month cache: {settings['dir']} (split {args.split}, shard {args.shard}, rebuild {args.rebuild})")
    cluster, failed_all = {}, []
    for name, data_params in (("train", train_params), ("test", test_params)):
        if args.split not in (name, "both"):
            continue
        start, minutes = time.perf_counter(), {}
        entries, failed = gates_lean.ensure_months_cached(
            data_params, parameters["variables"], datapath_args, background_params, settings["dir"],
            rebuild=args.rebuild or settings["rebuild"], verbose=verbose, shard=shard,
            loading_times_out=minutes, cluster=cluster)
        failed_all += failed
        print(f"{name}: {len(entries)} months cached = {sum(e.n_samples for e in entries)} samples; "
              f"{len(minutes)} loaded now in {(time.perf_counter() - start) / 60:.1f} min; failed: {failed}")
    if cluster.get("cluster") is not None:
        cluster["cluster"].close()
        cluster["client"].close()
    if failed_all:
        print(f"FLAG: {len(failed_all)} month(s) could not be loaded: {failed_all}")


if __name__ == "__main__":
    main()
