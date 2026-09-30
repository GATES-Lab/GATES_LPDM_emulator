---
html_theme.sidebar_secondary.remove: true
---

# GATES

**Graph-Neural-Network Atmospheric Transport Emulation System**

Fast footprints for greenhouse gas emissions inference from satellites.

::::{div} gates-buttons
```{button-ref} get_started/index
:ref-type: doc
:color: primary

Get started
```
```{button-link} https://gmd.copernicus.org/articles/19/1893/2026/
:color: secondary
:outline:

Read the paper
```
```{button-link} https://github.com/GATES-Lab/GATES_LPDM_emulator
:color: secondary
:outline:

{fab}`github` GitHub
```
::::

## What GATES does

GATES emulates a Lagrangian Particle Dispersion Model (LPDM). Given the meteorology around a satellite measurement, it predicts the **footprint**: a 2D (lat × lon) field describing how sensitive that measurement is to surface emissions.

Footprints are combined with flux maps and observations in atmospheric inversions to estimate greenhouse gas emissions. GATES produces them in about one second each instead of about ten minutes with the LPDM, which makes inversions using dense satellite data practical.

## Documentation

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`rocket` Get started
:link: get_started/index
:link-type: doc
:class-card: gates-card

Set up the environment and config, and launch your first training run.
:::

:::{grid-item-card} {fas}`book` User guide
:link: user_guide/index
:link-type: doc
:class-card: gates-card

Data formats, the parameter file, loss functions, evaluation, prediction and multi-region training.
:::

:::{grid-item-card} {fas}`code` API reference
:link: api/modules
:link-type: doc
:class-card: gates-card

Documentation of the `gates` package: data loading, the model, training and evaluation.
:::

::::

## The model

GATES is an encode–process–decode graph neural network, based on Keisler (2022) and GraphCast. It works on two levels: a lat-lon grid (the same shape for inputs and outputs) and a coarser hexagonal mesh built with the [`h3`](https://h3geo.org/) library.

```{figure} _static/gates_architecture.png
:alt: GATES architecture. Meteorology and static features on the lat-lon grid are encoded onto a hexagonal mesh, processed by four rounds of message passing, and decoded back onto the grid as a footprint.
:class: gates-figure

The GATES architecture. Inputs (meteorology at the observation time and 6 h and 12 h before, plus static features) are encoded from the grid onto the mesh, processed on the mesh, and decoded back onto the grid as a footprint. Figure 1 from [Fillola et al. (2026)](https://gmd.copernicus.org/articles/19/1893/2026/), CC BY 4.0.
```

1. **Encoder:** grid nodes send their features to nearby mesh nodes.
2. **Processor:** several rounds (`num_blocks`, 4 by default) of message passing on the mesh.
3. **Decoder:** each grid node collects from its nearest mesh nodes to predict the footprint value.

## Citation

If you use GATES, please cite:

```bibtex
@Article{gmd-19-1893-2026,
  AUTHOR  = {Fillola, E. and Santos-Rodriguez, R. and Tunnicliffe, R. and Clark, J. N. and Keshtmand, N. and Ganesan, A. and Rigby, M.},
  TITLE   = {Enabling fast greenhouse gas emissions inference from satellites with GATES: a Graph-Neural-Network Atmospheric Transport Emulation System},
  JOURNAL = {Geoscientific Model Development},
  VOLUME  = {19},
  YEAR    = {2026},
  NUMBER  = {5},
  PAGES   = {1893--1915},
  URL     = {https://gmd.copernicus.org/articles/19/1893/2026/},
  DOI     = {10.5194/gmd-19-1893-2026}
}
```

```{toctree}
:hidden:
:caption: Get Started

get_started/index
```

```{toctree}
:hidden:
:caption: User Guide

user_guide/index
```

```{toctree}
:hidden:
:caption: API Reference

api/modules
```
