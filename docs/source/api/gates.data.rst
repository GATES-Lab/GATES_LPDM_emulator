gates.data
==================

.. .. automodule:: gates.data
..    :members:
..    :show-inheritance:
..    :undoc-members:

Modules
-------

- :doc:`gates.data.load_data` -- Functions to load footprints, meteorology, and topography data, ready to feed into the GATES model or others.
- :doc:`gates.data.datasets` -- PyTorch ``Dataset``/``DataLoader`` classes (``FootprintDataset``, ``InputsDataset``) for feeding data to the model.
- :doc:`gates.data.load_data_helper_funs` -- Internal helper functions supporting data loading (e.g. distance calculations, file handling).

.. toctree::
   :maxdepth: 1
   :hidden:

   gates.data.load_data
   gates.data.datasets
   gates.data.load_data_helper_funs
