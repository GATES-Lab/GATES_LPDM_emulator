"""Processor for the latent graph

Functions for building GNN

Original code (GraphProcessor, build_satellite_processor_block) was developed by https://github.com/openclimatefix/graph_weather for weather forecasting applications. Here, setup is modified and used as part of the GATES model


author: Elena Fillola @elenafillo

"""

import torch
#import einops 

from .graph_net_block import GraphSatelliteProcessor


class SatelliteProcessor(torch.nn.Module):
    """Processor for latent graphD"""

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
        """
        Latent graph processor

        Args:
            input_dim: Input dimension for the node
            edge_dim: Edge input dimension
            num_blocks: Number of message passing blocks
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
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
        """
        Adds features to the encoding graph

        Args:
            x: Torch tensor containing node features
            edge_index: Connectivity of graph, of shape [2, Num edges] in COO format
            edge_attr: Edge attribues in [Num edges, Features] shape

        Returns:
            torch Tensor containing the values of the nodes of the graph
        """
        #print("processor", batch)
        #print(x.size(), edge_index.size())
        out, _ = self.graph_processor(x, edge_index, edge_attr, batch)

        return out
