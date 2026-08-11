# How to: SLURM jobs and Apptainer containers on Isambard-AI


| What | Where |
|---|---|
| Production container | `/projects/b5bn/data/env/gates_env_v2.sif` |
| Canonical definition file | [`gates_env.def`](../gates_env.def) (repo root) |
| Build job for the container | [`gates_env_v2_build.sbatch`](../gates_env_v2_build.sbatch) |

---

## 1. Start the container

The host mounts the project filesystem at `/lus/lfs1aip2/projects`, and `/projects` is only
a symlink to it. Symlinks don't resolve inside a container, so every invocation needs the
bind `--bind /lus,/lus/lfs1aip2/projects:/projects`: it mounts the real Lustre tree and
re-exposes it at `/projects`, making `config.yml`'s `/projects/...` paths work unchanged.

`--nv` passes the NVIDIA GPU + driver through. Harmless on a CPU-only node (prints a
warning), required wherever CUDA is used.

### Run one command (what batch jobs do)

```bash
apptainer exec --nv --bind /lus,/lus/lfs1aip2/projects:/projects \
  /projects/b5bn/data/env/gates_env_v2.sif \
  python -u run_dual_sweep.py parameter_template_dual_small.json
```

### Interactive shell inside the container

```bash
apptainer shell --nv --bind /lus,/lus/lfs1aip2/projects:/projects \
  /projects/b5bn/data/env/gates_env_v2.sif
```

Prompt becomes `Apptainer>`; `exit` leaves. Your home directory and current working
directory are visible by default.

---

## 2. Create a container

Two patterns, both used to make `gates_env_v2.sif`. A `.def` file is the recipe; `apptainer
build` turns it into a `.sif` image. Builds are heavy so you could submit it as a job on the compute node.

### Pattern A — full build from a definition file

[`gates_env.def`](../gates_env.def) is the canonical recipe: it adds additional packages to NVIDIA's PyTorch image (HAVING THIS IS IMPORTANT) and layers on the geo/scientific stack. Skeleton of its structure:

```text
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:25.05-py3     # base image pulled from NGC

%post                                       # runs once at build time, as root
    apt-get install ... libproj-dev libgeos-dev ...
    python3 -m pip install ... xarray==2025.1.2 dask==2024.5.0 ...
    python3 -m pip install --no-build-isolation torch-scatter

%environment                                # runs at every container start
    export LC_ALL=C

%labels                                     # free-form metadata
    Base nvcr.io/nvidia/pytorch:25.05-py3
```

Build it exactly as [`gates_env_v2_build.sbatch`](../gates_env_v2_build.sbatch) does —
a CPU batch job (16 CPUs, 360 G, 4 h) whose whole payload is one line:

```bash
apptainer build --fakeroot /projects/b5bn/data/env/gates_env_v2.sif gates_env.def
```

`--fakeroot` lets an unprivileged user act as root inside the build (required on
Isambard). Submit, then watch `output_logs/gates_env_v2_build_slurm_<ID>.log`.

### Pattern B — overlay build (extend an existing image)

To add a package without re-pulling the base and recompiling torch-scatter/cartopy
(~minutes instead of hours), bootstrap from the local image. Real example,
[`gates_env_h5.def`](../gates_env_h5.def):

```text
Bootstrap: localimage
From: /lus/lfs1aip2/projects/public/b5bn/Nawid/GATES_LPDM_emulator/gates_env.sif

%post
    python3 -m pip install --no-cache-dir h5netcdf h5py
```

```bash
apptainer build --fakeroot gates_env_h5.sif gates_env_h5.def
```

Though sometimes this does have issues so I would recommend building from scratch


### Rules that keep builds safe & reproducible

- **Never overwrite an image in use** — build to a *new* name (`_v2`, `_v3`), test, then
  switch launchers over. Old images stay as instant rollback.
- **Version the `.def` in git**, in the repo root, one recipe per image.
- **Lost the `.def`?** Every `.sif` embeds it — recover with:
  ```bash
  apptainer inspect --deffile /projects/b5bn/data/env/gates_env_v2.sif
  ```
  (This is how `gates_env.def` in this repo was restored.)
- **Pin versions** in `%post` (`dask==2024.5.0`) — an unpinned rebuild six months later is
  a different environment.

---

## 3. Launch a job

`sbatch` submits a script to the SLURM queue to run on a compute node. The file extension
doesn't matter (`.sh`, `.sbatch`, ...) — what matters is the `#SBATCH` header lines at the
top of the script, which request the resources (partition, account, CPUs, memory, GPU,
wall time). 

### Dual model with Zarr meteorology (this branch)

[`launch_dual_zarr_isambard.sh`](../launch_dual_zarr_isambard.sh) trains the dual-head
model reading Zarr met stores from `/projects/b5bn/data/met_archive_zarr`:

```bash
sbatch launch_dual_zarr_isambard.sh                       # default parameter file
PARAM_FILE=my_params.json sbatch launch_dual_zarr_isambard.sh   # override
```

It defaults to `gates_env_v3.sif` (v2 + editable `gates` install; override with `SIF=...`)
and to [`parameter_dual_zarr_isambard.json`](../parameter_files/parameter_dual_zarr_isambard.json),
which points met loading at the Zarr archive via its `"data_dirs"` key. Do **not** repoint
`config.yml`'s `met_datadir` at the Zarr archive globally — other branches' NetCDF loaders
still read `met_archive` through it. Data constraint (as of 2026-08-11): SOUTHAMERICA is the
only region with Zarr met (2014–2017) *and* footprints (2012–2015) *and* CAMS backgrounds
(2013–2017), so usable years are **2014 and 2015** (the default file trains 2015, tests 2014).
