"""
Functions for building GNN

Original code (GraphProcessor, build_processor_block) was developed by https://github.com/openclimatefix/graph_weather for weather forecasting applications (Copyright (c) 2022 Open Climate Fix), with some of their code taken from https://github.com/CCSI-Toolset/MGN, available under the
US Government License


author: Elena Fillola @elenafillo

"""


from typing import Tuple, Optional
import einops

import torch
from torch import cat, nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_sum, scatter_mean
from torch.utils.checkpoint import checkpoint
import torch_geometric
import numpy as np

def weights_init_normal_rule(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = (1.0/np.sqrt(n))
        torch.nn.init.normal_(m.weight, 0, y)

def weights_init_uniform(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        torch.nn.init.uniform_(m.weight, 0,1)

def weights_init_uniform_rule(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        torch.nn.init.uniform_(m.weight, -y, y)

def weights_init_ones(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.ones_(m.weight)

def weights_init_xavier_uniform(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def weights_init_xavier_normal_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def weights_init_kaiming_uniform_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)

def weights_init_kaiming_normal_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

def bias_init_zeros(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.bias)

def bias_init_uniform_rule(m):
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        torch.nn.init.uniform_(m.bias, -y, y)

class MLP(nn.Module):
    """MLP for graph processing"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: Optional[str] = "LayerNorm",
        use_checkpointing: bool = False,
        dropout: Optional[float] = 0.,
        final_activation: Optional[str] = None,
    ):
        """
        MLP

        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type one of 'LayerNorm', 'GraphNorm',
                'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Whether to use gradient checkpointing or not
        """
        super().__init__()
        #super(MLP, self).__init__()
        #super(nn.Module, self).__init__()

        #print(f"MLP - dropout {dropout}")
        self.use_checkpointing = use_checkpointing

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        """
        if dropout>0:
            layers.append(nn.Dropout(p=dropout))
        for _ in range(hidden_layers - 1):
            if dropout==0:
                #print("not adding dropout")
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            else:
                #print("adding dropout!")
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)]
        """
        for _ in range(hidden_layers - 1):
            if dropout==0:
                #print("not adding dropout")
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            else:
                #print("adding dropout!")
                layers += [nn.Dropout(p=dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        if final_activation is not None:
            print("adding ", final_activation)
            activations = {"ReLU":nn.ReLU(), "Sigmoid":nn.Sigmoid(), "LReLU":nn.LeakyReLU()}
            layers.append(activations[final_activation])

        if norm_type is not None:
            norm_dict = {"LayerNorm": nn.LayerNorm, 
                         "GraphNorm":torch_geometric.nn.norm.graph_norm.GraphNorm,
                         "InstanceNorm": nn.InstanceNorm1d,
                         "BatchNorm" : nn.BatchNorm1d,
                         "MessageNorm": torch_geometric.nn.norm.msg_norm.MessageNorm
                         }
            """assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]"""

            assert norm_type in norm_dict.keys(), "norm type not recognised"

            #norm_layer = getattr(nn, norm_type)
            layers.append(norm_dict[norm_type](out_dim))

        self.model = nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the MLP

        Args:
            x: Node or edge features

        Returns:
            The transformed tensor
        """
        if self.use_checkpointing:
            out = checkpoint(self.model, x, use_reentrant=False)
        else:
            out = self.model(x)
        return out


#############################

# issue with MessagePassing class:
# Only node features are updated after MP iterations
# Need to use MetaLayer to also allow edge features to update


class EdgeProcessor(nn.Module):
    """EdgeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float = 0, batched_data=False):
        """
        Edge processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layers
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        #super(EdgeProcessor, self).__init__()
        #super(nn.Module, self).__init__()
        self.edge_mlp = MLP(
            2 * in_dim_node + in_dim_edge, in_dim_edge, hidden_dim, hidden_layers, norm_type, dropout=dropout)
        self.batched_data=batched_data
        #print("in edge", self.batched_data)

    def forward(
        self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor,batch, u=None,
    ) -> torch.Tensor:
        """
        Compute the edge part of the message passing

        Args:
            src: Source node tensor
            dest: Destination node tensor
            edge_attr: Edge attributes
            u: Global attributes, ignored

        Returns:
            The updated edge attributes
        """
        #print("in edge processor")
        #print("edge processor", src.size(), dest.size(), edge_attr.size())
        #print("edge batch", batch)
        #print("edge processor", src.size(), dest.size(), edge_attr.size())
        out = cat(
            [src, dest, edge_attr], -1
        )  # concatenate source node, destination node, and edge embeddings
        
        #print("doing mlp")
        #print(self.edge_mlp)
        #print(out.size())
        out = self.edge_mlp(out)
        #print("mlp done")
        out += edge_attr  # residual connection

        return out


class NodeProcessor(nn.Module):
    """NodeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float=0,
    ):
        """
        Node Processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        #super(NodeProcessor, self).__init__()
        #super(nn.Module, self).__init__()
        self.node_mlp = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type, dropout=dropout
        )

        # batched_data not needed because edge_attr is changed in edge if needed
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch, u=None,
    ) -> torch.Tensor:
        """
        Compute the node feature updates in message passing

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch IDX, ignored

        Returns:
            torch.Tensor with updated node attributes
        """
        #print("node batch", batch)
        row, col = edge_index
        #print("scatter in graph", edge_index[1,:])

        if batch is not None:
            #print("rearranging")
            edge_attr = einops.rearrange(edge_attr, "(b n) f -> b n f", b=batch)

        out = scatter_sum(edge_attr, col, dim=0)  # aggregate edge message by target
        #(out.size(), x.size())
        out = cat([x, out], dim=-1)
        out = self.node_mlp(out)
        out += x  # residual connection
        #print(out.size())
        return out

class EdgeSatelliteProcessor(nn.Module):
    """EdgeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float = 0, batched_data=False):
        """
        Edge processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layers
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        #super(EdgeProcessor, self).__init__()
        #super(nn.Module, self).__init__()
        self.edge_mlp = MLP(
            2 * in_dim_node + in_dim_edge, in_dim_edge, hidden_dim, hidden_layers, norm_type, dropout=dropout)
        self.batched_data=batched_data
        #print("in edge", self.batched_data)

    def forward(
        self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor,batch, u=None,
    ) -> torch.Tensor:
        """
        Compute the edge part of the message passing

        Args:
            src: Source node tensor
            dest: Destination node tensor
            edge_attr: Edge attributes
            u: Global attributes, ignored

        Returns:
            The updated edge attributes
        """
        #print("in edge processor")
        #print("edge processor", src.size(), dest.size(), edge_attr.size())
        #print("edge batch", batch)
        #print("edge processor", src.size(), dest.size(), edge_attr.size())
        out = cat(
            [src, dest, edge_attr], -1
        )  # concatenate source node, destination node, and edge embeddings
        
        #print("doing mlp")
        #print(self.edge_mlp)
        #print(out.size())
        out = self.edge_mlp(out)
        #print("mlp done")
        #out += edge_attr  # residual connection

        return out


def scatter_cat(values, indeces):
    # Determine the maximum index value
    max_index = torch.max(indeces)

    n_neighbours = 7
    
    # Create a tensor to hold all scattered values
    scattered_values = torch.zeros((max_index + 1, n_neighbours*values[0].shape[-1]))
    #print(values.size())
    # Scatter the values into the tensor
    for n, idx in enumerate(np.unique(indeces)):
        if len(torch.where(indeces==idx)[0]) == n_neighbours:
            #print("in 372")
            #print(torch.where(indeces==idx)[0])
            #print(values[torch.where(indeces==idx)[0],:])
            #print(values[torch.where(indeces==idx)[0],:].size())
            scattered_values[n, :] = torch.flatten(values[torch.where(indeces==idx)[0],:])
            #print(scattered_values)
        else:
            #print("in 378")
            #print(((n_neighbours-len(torch.where(indeces==idx)[0]))*values[0].shape[-1], len(torch.where(indeces==idx)[0])))
            #print(torch.where(indeces==idx)[0])
            #print(values[torch.where(indeces==idx)[0],:])
            missing = torch.zeros(((n_neighbours-len(torch.where(indeces==idx)[0]))*values[0].shape[-1]))
            scattered_values[n, :] = torch.cat([torch.flatten(values[torch.where(indeces==idx)[0],:]), missing])
    return scattered_values    
    

def scatter_cat_v2(values, indeces, idx_positions=None, chunk_length=None):
    # Determine the maximum index value

    if idx_positions is None:
        idx_positions = []
        chunk_length = []
        for n, idx in enumerate(np.unique(indeces)):
            idx_positions.append(torch.where(indeces==idx)[0].tolist())
            chunk_length.append(len(torch.where(indeces==idx)[0].tolist()))
        idx_positions = [item for row in idx_positions for item in row]


    # Create a tensor to hold all scattered values
    scattered_values = values[idx_positions,:]
    scattered_values = torch.split(scattered_values, chunk_length)
    scattered_values = pad_sequence(scattered_values, batch_first=True)
    scattered_values = torch.reshape(scattered_values, (scattered_values.size()[0],scattered_values.size()[1]*scattered_values.size()[2]))

    #print(idx_positions,chunk_length)

    return scattered_values  


class NodeSatelliteProcessorDisaggregated(nn.Module):
    """NodeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float=0,
    ):
        """
        Node Processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        #super(NodeProcessor, self).__init__()
        #super(nn.Module, self).__init__()

        ## scatter isnt used, only here for continuity


        # MLP1 takes the node features of the origin node and the edge features
        self.node_mlp_1 = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type, dropout=dropout
        )
        # MLP 2 takes the DISaggregated outputs and the node features at destination node
        self.node_mlp_2 = MLP(
            7*in_dim_node, in_dim_node, 7*hidden_dim, hidden_layers, norm_type, dropout=dropout
        )

        # MLP 2 takes the DISaggregated outputs and the node features at destination node
        self.node_mlp_3 = MLP(
            2*in_dim_node, in_dim_node, hidden_dim, hidden_layers, norm_type, dropout=dropout
        )

        print("disagg")
        print("using scatter cat v2")
        # batched_data not needed because edge_attr is changed in edge if needed
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch, u=None,
    ) -> torch.Tensor:
        """
        Compute the node feature updates in message passing

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch IDX, ignored

        Returns:
            torch.Tensor with updated node attributes
        """
        #print("node batch", batch)
        #print("here1")
        row, col = edge_index
        #print(edge_index)
        #print(hasattr(self, "idx_positions"))
        if not hasattr(self, "idx_positions") or len(self.idx_positions)!=len(col):
            idx_positions = []
            chunk_length = []
            for n, idx in enumerate(np.unique(col)):
                idx_positions.append(torch.where(col==idx)[0].tolist())
                chunk_length.append(len(torch.where(col==idx)[0].tolist()))
            idx_positions = [item for row in idx_positions for item in row]
            print("did this loop")
        #print(x.size())
            self.idx_positions = idx_positions
            self.chunk_length = chunk_length

        out = torch.cat([x[row], edge_attr], dim=1)
        #print("mlp 1")
        #print(out.size())
        out = self.node_mlp_1(out)
        #print("here2")
        #print(out.size())
        #out2 = einops.rearrange(out, "(b n) f -> b n f", b=5)
        #print(out2.size())
        #out = scatter_mean(out, col, dim=0)  # aggregate edge message by target
        
        out = scatter_cat_v2(out, col, self.idx_positions, self.chunk_length)
        #print(col.size())
        #print(out.size())
        out = self.node_mlp_2(out)

        #print(out.size(), x.size())
        out = cat([x, out], dim=-1)
        #print(out.size(), x.size())
        #print("mlp 2")
        out = self.node_mlp_3(out)

        #print(out.size(), x.size())
        #out += x  # residual connection
        #print(out.size())
        return out



class NodeSatelliteProcessor(nn.Module):
    """NodeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float=0,
        scatter: str="mean",
    ):
        """
        Node Processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        #super(NodeProcessor, self).__init__()
        #super(nn.Module, self).__init__()

        # MLP1 takes the node features of the origin node and the edge features
        # outputs are aggregated by destination node
        self.node_mlp_1 = MLP(
            in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type, dropout=dropout
        )
        # MLP 2 takes the aggregated outputs and the node features at destination node
        self.node_mlp_2 = MLP(
            2*in_dim_node, in_dim_node, hidden_dim, hidden_layers, norm_type, dropout=dropout
        )

        self.scatter=scatter
        print("hello")


        # batched_data not needed because edge_attr is changed in edge if needed
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch, u=None,
    ) -> torch.Tensor:
        """
        Compute the node feature updates in message passing

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch IDX, ignored

        Returns:
            torch.Tensor with updated node attributes
        """
        #print("node batch", batch)
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        #print("mlp 1")

        out = self.node_mlp_1(out)

        #print(out.size(), edge_index.size())
        if self.scatter=="sum":
            out = scatter_sum(out, col, dim=0)  # aggregate edge message by target
        if self.scatter=="mean":
            out = scatter_mean(out, col, dim=0)  # aggregate edge message by target
        #print(out.size(), x.size())
        out = cat([x, out], dim=-1)
        #print(out.size(), x.size())
        #print("mlp 2")
        out = self.node_mlp_2(out)

        #print(out.size(), x.size())
        #out += x  # residual connection
        #print(out.size())
        return out

class NodeSatelliteProcessorAttention(nn.Module):
    """NodeProcessor"""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float=0,
        scatter: str="mean", attention_mask=None
    ):
        """
        Node Processor

        Args:
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim: Number of nodes in hidden layer
            hidden_layers: Number of hidden layers
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        #super(NodeProcessor, self).__init__()
        #super(nn.Module, self).__init__()

        # MLP1 takes the node features of the origin node and the edge features
        # outputs are aggregated by destination node
        #self.node_mlp_1 = MLP(
        #    in_dim_node + in_dim_edge, in_dim_node, hidden_dim, hidden_layers, norm_type, dropout=dropout
        #)

        # note that in BluePebble's torch version multihead attention returns averaged attention weights across all heads (which removes the idea that different heads will pay attention to different components). Future torch versions resolve this by adding average_attn_weights=False to self.attn_layer.forward (which returns disaggregated weights). best way to get around it in current version is to have multiple single-head attention layers
        self.attn_layer = torch.nn.MultiheadAttention(in_dim_node, 1)


        # MLP 2 takes the aggregated outputs and the node features at destination node
        self.node_mlp_2 = MLP(
            2*in_dim_node, in_dim_node, hidden_dim, hidden_layers, norm_type, dropout=dropout
        )

        print("attention!")
        self.attention_mask = attention_mask

        ## approach
        # nodes_only - average nodes by attention weight
        self.approach="nodes_only"


        # batched_data not needed because edge_attr is changed in edge if needed
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch, u=None,
    ) -> torch.Tensor:
        """
        Compute the node feature updates in message passing

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes
            u: Global attributes, ignored
            batch: Batch IDX, ignored

        Returns:
            torch.Tensor with updated node attributes
        """
        #print("node batch", batch)
        self.attention_mask = self.attention_mask.bool().to(x.device)
        #print(self.attention_mask.device)

        row, col = edge_index
        #print(x.size(), self.attention_mask.size())  
        if batch is not None:
            #print("rearranging")
            x = einops.rearrange(x, "(b n) f -> n b f", b=batch)

        #print(x.size(), self.attention_mask.size())        
        #print(batch)
        attn_output, attn_output_weights = self.attn_layer(x, x, x, attn_mask=self.attention_mask)

        #print(x)
        #print(attn_output.size(), attn_output_weights.size())
        #print(attn_output_weights, attn_output_weights.size())
        
        if batch is not None:
            #print("rearranging back")
            x = einops.rearrange(x, "n b f -> (b n) f", b=batch)
            attn_output = einops.rearrange(attn_output, "n b f -> (b n) f", b=batch)

            #print(attn_output.size())
            
        if self.approach=="nodes_only":
            out = self.node_mlp_2(cat([x, attn_output], dim=-1))
        else:
            print("not implemented! how did you even get here")

        # reshape into batches

        #out = torch.cat([x[row], edge_attr], dim=1)
        #print("mlp 1")
        #out = self.node_mlp_1(out)
        #print(np.shape(out))
        #out = scatter_sum(out, col, dim=0)  # aggregate edge message by target
        #print(np.shape(out), np.shape(x))
        #print(col, torch.where(col==3))
        #out = cat([x, out], dim=-1)
        #print(out.size(), x.size())
        #print("mlp 2")
        #out = self.node_mlp_2(out)

        #print(out.size(), x.size())
        #out += x  # residual connection
        #print(out.size())
        return out


def build_satellite_graph_processor_block(
    in_dim_node: int = 128,
    in_dim_edge: int = 128,
    hidden_dim_node: int = 128,
    hidden_dim_edge: int = 128,
    hidden_layers_node: int = 2,
    hidden_layers_edge: int = 2,
    norm_type: str = "LayerNorm",
    dropout: float=0,
    scatter: str="mean",disaggregated=False, attention=False,attention_mask=None
) -> torch.nn.Module:
    """
    Build the Graph Net Block

    Args:
        in_dim_node: Input node feature dimension
        in_dim_edge: Input edge feature dimension
        hidden_dim_node: Number of nodes in hidden layer for graph node processing
        hidden_dim_edge: Number of nodes in hidden layer for graph edge processing
        hidden_layers_node: Number of hidden layers for node processing
        hidden_layers_edge: Number of hidden layers for edge processing
        norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
    Returns:
        torch.nn.Module for the graph processing block
    """
    #print("build", batched_data)
    if disaggregated:
        return MetaLayer(
            edge_model=EdgeSatelliteProcessor(
                in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type, dropout=dropout
            ),
            node_model=NodeSatelliteProcessorDisaggregated(
                in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type, dropout=dropout
            ),
        )
    elif attention:
        assert attention_mask is not None, "Neighbours attention mask should have been passed!"
        return MetaLayer(
            edge_model=EdgeSatelliteProcessor(
                in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type, dropout=dropout
            ),
            node_model=NodeSatelliteProcessorAttention(
                in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type, dropout=dropout, scatter=scatter, attention_mask=attention_mask
            ),
        )        
    else:
        return MetaLayer(
            edge_model=EdgeSatelliteProcessor(
                in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type, dropout=dropout
            ),
            node_model=NodeSatelliteProcessor(
                in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type, dropout=dropout, scatter=scatter
            ),
        )
            
def build_graph_processor_block(
    in_dim_node: int = 128,
    in_dim_edge: int = 128,
    hidden_dim_node: int = 128,
    hidden_dim_edge: int = 128,
    hidden_layers_node: int = 2,
    hidden_layers_edge: int = 2,
    norm_type: str = "LayerNorm",
    dropout: float=0
) -> torch.nn.Module:
    """
    Build the Graph Net Block

    Args:
        in_dim_node: Input node feature dimension
        in_dim_edge: Input edge feature dimension
        hidden_dim_node: Number of nodes in hidden layer for graph node processing
        hidden_dim_edge: Number of nodes in hidden layer for graph edge processing
        hidden_layers_node: Number of hidden layers for node processing
        hidden_layers_edge: Number of hidden layers for edge processing
        norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
    Returns:
        torch.nn.Module for the graph processing block
    """
    #print("build", batched_data)
    return MetaLayer(
        edge_model=EdgeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_edge, hidden_layers_edge, norm_type, dropout=dropout
        ),
        node_model=NodeProcessor(
            in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type, dropout=dropout
        ),
    )


class GraphProcessor(nn.Module):
    """Overall graph processor"""

    def __init__(
        self,
        mp_iterations: int = 15,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim_node: int = 128,
        hidden_dim_edge: int = 128,
        hidden_layers_node: int = 2,
        hidden_layers_edge: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float = 0,
    ):
        """
        Graph Processor

        Args:
            mp_iterations: number of message-passing iterations (graph processor blocks)
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim_node: Number of nodes in hidden layers for node processing
            hidden_dim_edge: Number of nodes in hidden layers for edge processing
            hidden_layers_node: Number of hidden layers for node processing
            hidden_layers_edge: Number of hidden layers for edge processing
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        #super(GraphProcessor, self).__init__()
        #super(nn.Module, self).__init__()
        #print("beginning")
        self.blocks = nn.ModuleList()
        for _ in range(mp_iterations):
            #print("loop", mp_iterations)
            self.blocks.append(
                build_graph_processor_block(
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    norm_type, dropout=dropout
                )
            )

        #print("set up")
        #print("after loop", self.batched_data)
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute updates to the graph in message passing method

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes

        Returns:
            Updated nodes and edge attributes
        """
        #print("graph block", batch)
        for block in self.blocks:
            #print("in graph processor")
            #print(x.size(), edge_index.size(), edge_attr.size())
            #print("in block")
            #print(edge_index[0])
            x, edge_attr, _ = block(x, edge_index, edge_attr, batch)#, batch=torch.tensor(np.arange(x.size()[0])))                
        return x, edge_attr

class GraphSatelliteProcessor(nn.Module):
    """Overall graph processor"""

    def __init__(
        self,
        mp_iterations: int = 15,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim_node: int = 128,
        hidden_dim_edge: int = 128,
        hidden_layers_node: int = 2,
        hidden_layers_edge: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float = 0,
        scatter: str="mean", disaggregated=False, attention=False, attention_mask=None
    ):
        """
        Graph Processor

        Args:
            mp_iterations: number of message-passing iterations (graph processor blocks)
            in_dim_node: Input node feature dimension
            in_dim_edge: Input edge feature dimension
            hidden_dim_node: Number of nodes in hidden layers for node processing
            hidden_dim_edge: Number of nodes in hidden layers for edge processing
            hidden_layers_node: Number of hidden layers for node processing
            hidden_layers_edge: Number of hidden layers for edge processing
            norm_type: Normalization type
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        """
        super().__init__()
        #super(GraphProcessor, self).__init__()
        #super(nn.Module, self).__init__()
        #print("beginning")
        self.blocks = nn.ModuleList()
        print("in graph processor!")
        for _ in range(mp_iterations):
            #print("loop", mp_iterations)
            self.blocks.append(
                build_satellite_graph_processor_block(
                    in_dim_node,
                    in_dim_edge,
                    hidden_dim_node,
                    hidden_dim_edge,
                    hidden_layers_node,
                    hidden_layers_edge,
                    norm_type, dropout=dropout, scatter=scatter,disaggregated=disaggregated, attention=attention, attention_mask=attention_mask
                )
            )

        #print("set up, satellite version")
        #print("after loop", self.batched_data)
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute updates to the graph in message passing method

        Args:
            x: Input nodes
            edge_index: Edge indicies in COO format
            edge_attr: Edge attributes

        Returns:
            Updated nodes and edge attributes
        """
        #print("graph block", batch)
        for block in self.blocks:
            #print("in graph processor")
            #print(x.size(), edge_index.size(), edge_attr.size())
            #print("in block")
            #print(edge_index[0])
            x, edge_attr, _ = block(x, edge_index, edge_attr, batch)#, batch=torch.tensor(np.arange(x.size()[0])))                
        
        return x, edge_attr




