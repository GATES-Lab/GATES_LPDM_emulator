# Working on the Documentation

This project's documentation is built with [Sphinx](https://www.sphinx-doc.org/) using the [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/). It lives under `docs/`, separate from the main `gates_env` used for training/running the model — you don't need PyTorch, CUDA, or any of the other heavy dependencies installed to build the docs.

---

## 1. Install the Docs Environment

The docs environment is defined in `env_gates_docs.yml` and only contains Sphinx and its extensions:

```bash
conda env create -f env_gates_docs.yml
conda activate gates_docs
```

Then install `gates` itself, without its runtime dependencies (numpy, torch, xarray, etc. are mocked out in `conf.py` so Sphinx never needs them):

```bash
pip install --no-deps -e .
```

---

## 2. Launch the Docs Locally (Live-Reload)

From the `docs/` folder, run:

```bash
cd docs
sphinx-autobuild source build/html --watch ../gates
```

This builds the site, serves it at [http://127.0.0.1:8000](http://127.0.0.1:8000), and automatically rebuilds and refreshes your browser whenever you edit a page under `source/` or a docstring under `gates/`.

For a one-off build without live-reload:

```bash
cd docs
make html
```

The output lands in `docs/build/html/index.html`.

---

## 3. Where Things Live

- `source/conf.py` — Sphinx configuration (theme, extensions, mocked imports).
- `source/index.rst` — landing page and top-level navigation.
- `source/api/` — API reference pages. Each page pairs an `automodule` directive with a hand-written "Modules" summary list, plus a hidden `toctree` for sidebar navigation.

**Note:** `source/api/*.rst` were originally generated with `sphinx-apidoc` and then hand-edited (titles, ordering, module summaries). Re-running `sphinx-apidoc -f` will overwrite these edits — add new module pages by hand following the existing pattern instead.

---

## To-Do

- [ ] Add `docs/build/` and `docs/source/_autosummary/` to `.gitignore`.
- [ ] Write real landing-page content for `index.rst` (currently placeholder).
- [ ] Wire the `How_Tos/` guides into the docs nav (via `myst_parser`, keeping `How_Tos/` as the source of truth).
- [ ] Double-check the one-line module summaries in `source/api/` against what each file actually does — some were written from sparse docstrings and may need correcting.
- [ ] Add navbar polish: GitHub repo link/icon, logo (`html_theme_options` in `conf.py`).
- [ ] Set up a GitHub Actions workflow to build and publish to `gh-pages` once ready to deploy.
