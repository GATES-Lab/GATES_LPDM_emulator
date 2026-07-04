"""Processor for the latent graph

Functions for building GNN

Original code (GraphProcessor, build_satellite_processor_block) was developed by https://github.com/openclimatefix/graph_weather for weather forecasting applications. Here, setup is modified and used as part of the GATES model


author: Elena Fillola @elenafillo

"""

import torch
#import einops 

from .graph_net_block import GraphSatelliteProcessor


class SatelliteProcessor(torch.nn.Module):
    """Processor for the latent graph."""

    def __init__(
        self,
        input_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        mlp_norm_type: str = "LayerNorm",
        dropout: float=0,
        scatter: str="mean",disaggregated=False, attention=False,attention_mask=None
    ):
        """Initialize the latent graph processor.

        Args:
            input_dim (int, optional): Input dimension for the node. Defaults to 256.
            edge_dim (int, optional): Edge input dimension. Defaults to 256.
            num_blocks (int, optional): Number of message passing blocks. Defaults to 9.
            hidden_dim_processor_node (int, optional): Hidden dimension of the node
                processors. Defaults to 256.
            hidden_dim_processor_edge (int, optional): Hidden dimension of the edge
                processors. Defaults to 256.
            hidden_layers_processor_node (int, optional): Number of hidden layers in
                the node processors. Defaults to 2.
            hidden_layers_processor_edge (int, optional): Number of hidden layers in
                the edge processors. Defaults to 2.
            mlp_norm_type (str, optional): Type of norm for the MLPs, one of
                "LayerNorm", "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm",
                or None. Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
            scatter (str, optional): Scatter reduction to use for message passing
                (e.g. "mean"). Defaults to "mean".
            disaggregated (bool, optional): <FILL IN>. Defaults to False.
            attention (bool, optional): <FILL IN>. Defaults to False.
            attention_mask (optional): <FILL IN>. Defaults to None.
        """
        super().__init__()
        # Build the default graph
        # Take features from encoder and put into processor graph
        self.input_dim = input_dim
        print("set up graph")
        self.graph_processor = GraphSatelliteProcessor(
            num_blocks,
            input_dim,
            edge_dim,
            hidden_dim_processor_node,
            hidden_dim_processor_edge,
            hidden_layers_processor_node,
            hidden_layers_processor_edge,
            mlp_norm_type, dropout=dropout, scatter=scatter, disaggregated=disaggregated, attention=attention,attention_mask=attention_mask
        )

    def forward(self, x: torch.Tensor, edge_index, edge_attr, batch=None) -> torch.Tensor:
        """Add features to the encoding graph by running the message-passing blocks.

        Args:
            x (torch.Tensor): Node features.
            edge_index: Connectivity of graph, of shape [2, Num edges] in COO format.
            edge_attr: Edge attributes in [Num edges, Features] shape.
            batch (optional): <FILL IN>. Defaults to None.

        Returns:
            torch.Tensor: Values of the nodes of the graph.
        """
        #print("processor", batch)
        #print(x.size(), edge_index.size())
        out, _ = self.graph_processor(x, edge_index, edge_attr, batch)

        return out
