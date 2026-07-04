gates
=============

.. .. automodule:: gates
..    :members:
..    :show-inheritance:
..    :undoc-members:

Modules
-------

- :doc:`gates.data` -- Loading and dataset utilities for footprint, meteorology, and topography data.
- :doc:`gates.model` -- The GATES GNN model architecture (encoder/processor/decoder) and forecaster.
- :doc:`gates.training` -- Training loop, dataclasses, and helper functions for training the model.
- :doc:`gates.evaluation` -- Metrics, loss functions, and post-processing for evaluating model predictions.
- :doc:`gates.plotting` -- Plotting utilities for visualising footprints and predictions.
- :doc:`gates.utils` -- Shared low-level utility functions.
- :doc:`gates.config` -- Loading and caching the project configuration (paths, domains, etc.).

.. toctree::
   :maxdepth: 1
   :hidden:

   gates.data
   gates.model
   gates.training
   gates.evaluation
   gates.plotting
   gates.utils
   gates.config
