gates.model.layers
==========================

.. automodule:: gates.model.layers
   :members:
   :show-inheritance:
   :undoc-members:

Modules
-------

- :doc:`gates.model.layers.encoder` -- Encodes the input graph into the latent graph representation.
- :doc:`gates.model.layers.processor` -- Builds the GNN message-passing blocks that process the latent graph.
- :doc:`gates.model.layers.decoder` -- Decodes the processed latent graph back into the original graph with updated values.
- :doc:`gates.model.layers.graph_net_block` -- Core GNN building blocks (MLP, message-passing blocks) shared by the encoder/processor/decoder.

.. toctree::
   :maxdepth: 1
   :hidden:

   gates.model.layers.encoder
   gates.model.layers.processor
   gates.model.layers.decoder
   gates.model.layers.graph_net_block
