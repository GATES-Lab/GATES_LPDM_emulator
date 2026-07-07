# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GATES'
copyright = '2026, Elena Fillola'
author = 'Elena Fillola'
release = 'v1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = []

# gates' runtime dependencies are intentionally not installed in the docs
# environment (see environment-docs.yml) to keep it lightweight. Mock them
# out so autodoc can still import gates/model to pull docstrings/signatures.
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "torch_scatter",
    "torchvision",
    "torchaudio",
    "numpy",
    "xarray",
    "pandas",
    "scipy",
    "sklearn",
    "dask",
    "xbatcher",
    "matplotlib",
    "cartopy",
    "einops",
    "h3",
    "joblib",
    "wandb",
    "yaml",
]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_favicon = '_static/favicon.ico'
html_theme_options = {
    "logo": {
        "text": "GATES",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/GATES-Lab/GATES_LPDM_emulator/tree/unify_training",
            "icon": "fa-brands fa-github",
        },
    ],
}
