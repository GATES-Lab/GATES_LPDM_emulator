# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import posixpath
import re
import sys
from pathlib import Path

from sphinx.util import logging

sys.path.insert(0, os.path.abspath("../.."))

logger = logging.getLogger(__name__)

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
            "url": "https://github.com/GATES-Lab/GATES_LPDM_emulator/tree/main",
            "icon": "fa-brands fa-github",
        },
    ],
}

# -- MyST ----------------------------------------------------------------------

# Generate GitHub-style anchors for headings up to ###, so that links such as
# HOW_TO_PARAMETER_FILE.md#variables have a target.
myst_heading_anchors = 3

# -- How_To link rewriting -------------------------------------------------------
#
# The guides in How_Tos/ are pulled into the site by one-line {include} stub
# pages. Their links are written for GitHub (HOW_TO_DATA.md, ../parameter_files/...),
# but inside an include they are resolved against the stub, where they don't
# exist. The hook below rewrites them in the included text only, so the How_Tos
# themselves keep working on GitHub:
#
#   HOW_TO_X.md[#anchor]  -> the stub page that includes HOW_TO_X.md
#   ../<repo path>        -> the file on GitHub
#   #anchor, http(s)://   -> unchanged
#
# When adding a How_To to the site, add its stub here too. Links to a How_To
# that is not listed raise a warning (an error under -W).

HOW_TO_PAGES = {
    "HOW_TO_CONFIG.md": "/get_started/config.md",
    "HOW_TO_WandB.md": "/get_started/wandb.md",
    "HOW_TO_LAUNCH.md": "/get_started/launch_training.md",
    "HOW_TO_DATA.md": "/user_guide/data_format.md",
    "HOW_TO_PARAMETER_FILE.md": "/user_guide/parameter_file.md",
    "HOW_TO_evaluation.md": "/user_guide/evaluation.md",
    "HOW_TO_LOSSES.md": "/user_guide/loss_functions.md",
    "HOW_TO_PREDICT.md": "/user_guide/predict.md",
    "HOW_TO_MULTIREGION.md": "/user_guide/multiregion.md",
}

GITHUB_BLOB = "https://github.com/GATES-Lab/GATES_LPDM_emulator/blob/main/"

REPO_ROOT = Path(__file__).resolve().parents[2]
HOW_TOS_DIR = REPO_ROOT / "How_Tos"

# ](target) or ](target "title"); the target has no spaces or closing brackets.
_LINK_RE = re.compile(r'\]\((?P<target>[^)\s]+)(?P<title>\s+"[^"]*")?\)')
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _rewrite_target(target, source_name, lineno):
    """Return the site version of one How_To link target."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("#"):
        return target  # URL, mailto:, or an anchor on the same page

    path, _, anchor = target.partition("#")
    anchor = f"#{anchor}" if anchor else ""
    name = posixpath.basename(path)

    if posixpath.dirname(posixpath.normpath(path)) in ("", ".") and name.startswith("HOW_TO_"):
        if name in HOW_TO_PAGES:
            return HOW_TO_PAGES[name] + anchor
        logger.warning(
            f"{source_name}:{lineno}: link to {name}, which has no page on the site. "
            f"Add a stub for it and list it in HOW_TO_PAGES in conf.py.",
            type="gates", subtype="how_to_link",
        )
        return target

    repo_path = posixpath.normpath(posixpath.join("How_Tos", path))
    if repo_path.startswith(".."):
        logger.warning(
            f"{source_name}:{lineno}: link {target!r} points outside the repository.",
            type="gates", subtype="how_to_link",
        )
        return target
    return GITHUB_BLOB + repo_path + anchor


def rewrite_how_to_links(app, relative_path, parent_docname, content):
    """``include-read`` handler: rewrite the links in an included How_To."""
    included = (Path(app.srcdir) / relative_path).resolve()
    if included.parent != HOW_TOS_DIR:
        return

    lines = content[0].split("\n")
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        lines[i] = _LINK_RE.sub(
            lambda m: "](" + _rewrite_target(m["target"], included.name, i + 1)
            + (m["title"] or "") + ")",
            line,
        )
    content[0] = "\n".join(lines)


def setup(app):
    app.connect("include-read", rewrite_how_to_links)
