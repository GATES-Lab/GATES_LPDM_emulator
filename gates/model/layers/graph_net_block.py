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
    """In-place init for ``nn.Linear`` weights: normal, mean 0, std ``1/sqrt(in_features)``.

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = (1.0/np.sqrt(n))
        torch.nn.init.normal_(m.weight, 0, y)

def weights_init_uniform(m):
    """In-place init for ``nn.Linear`` weights: uniform in [0, 1).

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        torch.nn.init.uniform_(m.weight, 0,1)

def weights_init_uniform_rule(m):
    """In-place init for ``nn.Linear`` weights: uniform in [-1/sqrt(in_features), 1/sqrt(in_features)).

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        n = m.in_features
        y = 1.0/np.sqrt(n)
        torch.nn.init.uniform_(m.weight, -y, y)

def weights_init_ones(m):
    """In-place init for ``nn.Linear`` weights: all ones.

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.ones_(m.weight)

def weights_init_xavier_uniform(m):
    """In-place Xavier/Glorot uniform init for ``nn.Linear`` weights.

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def weights_init_xavier_normal_(m):
    """In-place Xavier/Glorot normal init for ``nn.Linear`` weights.

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

def weights_init_kaiming_uniform_(m):
    """In-place Kaiming/He uniform init for ``nn.Linear`` weights.

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)

def weights_init_kaiming_normal_(m):
    """In-place Kaiming/He normal init for ``nn.Linear`` weights.

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)

def bias_init_zeros(m):
    """In-place init for ``nn.Linear`` biases: all zeros.

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.bias)

def bias_init_uniform_rule(m):
    """In-place init for ``nn.Linear`` biases: uniform in [-1/sqrt(in_features), 1/sqrt(in_features)).

    Intended to be passed to ``module.apply()``.

    Args:
        m (torch.nn.Module): Module being visited by ``apply()``; only acted on if
            it is an ``nn.Linear``.
    """
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
        """Initialize the MLP.

        Args:
            in_dim (int): Input dimension.
            out_dim (int, optional): Output dimension. Defaults to 128.
            hidden_dim (int, optional): Number of nodes in hidden layer. Defaults to 128.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            use_checkpointing (bool, optional): Whether to use gradient checkpointing
                or not. Defaults to False.
            dropout (float, optional): Dropout probability applied before each hidden
                layer (after the first) when > 0. Defaults to 0.
            final_activation (str, optional): Final activation to append, one of
                "ReLU", "Sigmoid", "LReLU", or None. Defaults to None.

        Raises:
            AssertionError: If ``norm_type`` is not a recognised normalization type.
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
        """Compute the MLP.

        Args:
            x (torch.Tensor): Node or edge features.

        Returns:
            torch.Tensor: The transformed tensor.
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
    """Edge processor: updates edge features from their source/destination nodes, with a residual connection."""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float = 0, batched_data=False):
        """Initialize the edge processor.

        Args:
            in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
            in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
            hidden_dim (int, optional): Number of nodes in hidden layers. Defaults to 128.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
            batched_data (bool, optional): <FILL IN> — accepted but not currently
                used in this class. Defaults to False.
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
        """Compute the edge part of the message passing.

        Args:
            src (torch.Tensor): Source node tensor.
            dest (torch.Tensor): Destination node tensor.
            edge_attr (torch.Tensor): Edge attributes.
            batch: <FILL IN> — accepted but not currently used in this method.
            u: Global attributes, ignored.

        Returns:
            torch.Tensor: The updated edge attributes.
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
    """Node processor: updates node features by aggregating incoming edge messages, with a residual connection."""

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float=0,
    ):
        """Initialize the node processor.

        Args:
            in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
            in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
            hidden_dim (int, optional): Number of nodes in hidden layer. Defaults to 128.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
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
        """Compute the node feature updates in message passing.

        Args:
            x (torch.Tensor): Input nodes.
            edge_index (torch.Tensor): Edge indices in COO format.
            edge_attr (torch.Tensor): Edge attributes.
            batch (int, optional): If not None, batch size used to reshape
                ``edge_attr`` to (batch, n_edges, features) before scattering.
            u: Global attributes, ignored.

        Returns:
            torch.Tensor: Updated node attributes.
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
    """Edge processor: updates edge features from their source/destination nodes.

    Unlike ``EdgeProcessor``, this does not apply a residual connection to the output.
    """

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float = 0, batched_data=False):
        """Initialize the edge processor.

        Args:
            in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
            in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
            hidden_dim (int, optional): Number of nodes in hidden layers. Defaults to 128.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
            batched_data (bool, optional): <FILL IN> — accepted but not currently
                used in this class. Defaults to False.
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
        """Compute the edge part of the message passing.

        Args:
            src (torch.Tensor): Source node tensor.
            dest (torch.Tensor): Destination node tensor.
            edge_attr (torch.Tensor): Edge attributes.
            batch: <FILL IN> — accepted but not currently used in this method.
            u: Global attributes, ignored.

        Returns:
            torch.Tensor: The updated edge attributes.
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
    """Concatenate feature vectors of entries sharing the same index into fixed-size (7-neighbour) rows.

    Assumes each index has at most ``n_neighbours=7`` entries; groups with fewer are
    zero-padded.

    Args:
        values (torch.Tensor): Feature vectors, shape [n_values, features].
        indeces (torch.Tensor): Group index for each entry in ``values``.

    Returns:
        torch.Tensor: Shape [max(indeces) + 1, 7 * features], each row the
        concatenated (and zero-padded) feature vectors for that group.
    """
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
    """Concatenate (and zero-pad) feature vectors of entries sharing the same index, without a fixed group size.

    Faster alternative to ``scatter_cat`` that groups and pads via
    ``torch.split``/``pad_sequence`` instead of a per-index Python loop over a
    fixed-size output. ``idx_positions``/``chunk_length`` can be precomputed once
    (e.g. via ``torch.where``/``np.unique`` on ``indeces``) and reused across calls
    with the same grouping to skip recomputation.

    Args:
        values (torch.Tensor): Feature vectors, shape [n_values, features].
        indeces (np.ndarray or torch.Tensor): Group index for each entry in ``values``.
        idx_positions (list[int], optional): Precomputed flat list of row positions
            in ``values``, grouped by index. If None, computed from ``indeces``.
            Defaults to None.
        chunk_length (list[int], optional): Precomputed group sizes matching
            ``idx_positions``. Required (and used) together with ``idx_positions``.
            Defaults to None.

    Returns:
        torch.Tensor: Shape [n_groups, max_group_size * features], each row the
        concatenated (and zero-padded) feature vectors for that group.
    """
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
    """Node processor variant that concatenates (rather than sum/mean-aggregates) incoming edge messages per node.

    Assumes each node has exactly 7 incoming edges (via ``scatter_cat_v2``).
    """

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float=0,
    ):
        """Initialize the node processor.

        Args:
            in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
            in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
            hidden_dim (int, optional): Number of nodes in hidden layer. Defaults to 128.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
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
        """Compute the node feature updates in message passing, via disaggregated (concatenated) edge messages.

        Args:
            x (torch.Tensor): Input nodes.
            edge_index (torch.Tensor): Edge indices in COO format.
            edge_attr (torch.Tensor): Edge attributes.
            batch: Batch IDX, ignored.
            u: Global attributes, ignored.

        Returns:
            torch.Tensor: Updated node attributes.
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
    """Node processor: updates node features by scatter-aggregating (sum or mean) incoming edge messages.

    Unlike ``NodeProcessor``, this does not apply a residual connection to the output.
    """

    def __init__(
        self,
        in_dim_node: int = 128,
        in_dim_edge: int = 128,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        norm_type: str = "LayerNorm",
        dropout: float=0,
        scatter: str="mean",
        residuals: bool = False,
    ):
        """Initialize the node processor.

        Args:
            in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
            in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
            hidden_dim (int, optional): Number of nodes in hidden layer. Defaults to 128.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
            scatter (str, optional): Aggregation to use for incoming edge messages,
                "sum" or "mean". Defaults to "mean".
            residuals (bool, optional): If True, add x back to node_mlp_2 output (residual update). Defaults to False.
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
        self.residuals=residuals
        print("hello")


        # batched_data not needed because edge_attr is changed in edge if needed
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch, u=None,
    ) -> torch.Tensor:
        """Compute the node feature updates in message passing.

        Args:
            x (torch.Tensor): Input nodes.
            edge_index (torch.Tensor): Edge indices in COO format.
            edge_attr (torch.Tensor): Edge attributes.
            batch: Batch IDX, ignored.
            u: Global attributes, ignored.

        Returns:
            torch.Tensor: Updated node attributes.
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

        if self.residuals:
            out += x  # residual connection
        return out

class NodeSatelliteProcessorAttention(nn.Module):
    """Node processor variant that updates node features using single-head self-attention instead of edge-based scatter aggregation.

    Note: ``edge_index``/``edge_attr`` are accepted for interface compatibility with
    the other node processors but are not used here — attention is computed over all
    nodes, masked by ``attention_mask`` to restrict it to connected neighbours.
    """

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
        """Initialize the node processor.

        Args:
            in_dim_node (int, optional): Input node feature dimension; also the
                attention embedding dimension. Defaults to 128.
            in_dim_edge (int, optional): <FILL IN> — accepted but not currently used
                in this class. Defaults to 128.
            hidden_dim (int, optional): Number of nodes in hidden layer. Defaults to 128.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
            scatter (str, optional): <FILL IN> — accepted but not currently used in
                this class. Defaults to "mean".
            attention_mask (torch.Tensor, optional): Boolean/float mask restricting
                attention to connected neighbours (see ``SatelliteEncoder``/
                ``SatelliteDynamicEncoder``). Defaults to None.
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
        """Compute the node feature updates via masked self-attention.

        Args:
            x (torch.Tensor): Input nodes.
            edge_index: Edge indices in COO format. Unused (see class docstring).
            edge_attr: Edge attributes. Unused (see class docstring).
            batch (int, optional): If not None, batch size used to reshape ``x`` for attention.
            u: Global attributes, ignored.

        Returns:
            torch.Tensor: Updated node attributes.
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
    scatter: str="mean",disaggregated=False, attention=False,attention_mask=None, residuals=False
) -> torch.nn.Module:
    """Build a satellite Graph Net Block (edge model + node model) as a ``MetaLayer``.

    Args:
        in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
        in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
        hidden_dim_node (int, optional): Number of nodes in hidden layer for graph
            node processing. Defaults to 128.
        hidden_dim_edge (int, optional): Number of nodes in hidden layer for graph
            edge processing. Defaults to 128.
        hidden_layers_node (int, optional): Number of hidden layers for node
            processing. Defaults to 2.
        hidden_layers_edge (int, optional): Number of hidden layers for edge
            processing. Defaults to 2.
        norm_type (str, optional): Normalization type, one of "LayerNorm",
            "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
            Defaults to "LayerNorm".
        dropout (float, optional): Dropout probability. Defaults to 0.
        scatter (str, optional): Aggregation to use for incoming edge messages in the
            (non-disaggregated, non-attention) node model, "sum" or "mean".
            Defaults to "mean".
        disaggregated (bool, optional): If True, uses
            ``NodeSatelliteProcessorDisaggregated`` for the node model (concatenates
            rather than aggregates incoming edge messages). Defaults to False.
        attention (bool, optional): If True (and ``disaggregated`` is False), uses
            ``NodeSatelliteProcessorAttention`` for the node model. Requires
            ``attention_mask``. Defaults to False.
        attention_mask (torch.Tensor, optional): Attention mask forwarded to
            ``NodeSatelliteProcessorAttention`` when ``attention=True``. Defaults to None.

    Returns:
        torch.nn.Module: ``MetaLayer`` for the graph processing block.

    Raises:
        AssertionError: If ``attention`` is True but ``attention_mask`` is None.
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
                in_dim_node, in_dim_edge, hidden_dim_node, hidden_layers_node, norm_type, dropout=dropout, scatter=scatter, residuals=residuals
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
    """Build a Graph Net Block (edge model + node model) as a ``MetaLayer``.

    Args:
        in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
        in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
        hidden_dim_node (int, optional): Number of nodes in hidden layer for graph
            node processing. Defaults to 128.
        hidden_dim_edge (int, optional): Number of nodes in hidden layer for graph
            edge processing. Defaults to 128.
        hidden_layers_node (int, optional): Number of hidden layers for node
            processing. Defaults to 2.
        hidden_layers_edge (int, optional): Number of hidden layers for edge
            processing. Defaults to 2.
        norm_type (str, optional): Normalization type, one of "LayerNorm",
            "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
            Defaults to "LayerNorm".
        dropout (float, optional): Dropout probability. Defaults to 0.

    Returns:
        torch.nn.Module: ``MetaLayer`` for the graph processing block.
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
    """Overall graph processor: a stack of message-passing Graph Net Blocks."""

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
        """Initialize the graph processor.

        Args:
            mp_iterations (int, optional): Number of message-passing iterations
                (graph processor blocks). Defaults to 15.
            in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
            in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
            hidden_dim_node (int, optional): Number of nodes in hidden layers for
                node processing. Defaults to 128.
            hidden_dim_edge (int, optional): Number of nodes in hidden layers for
                edge processing. Defaults to 128.
            hidden_layers_node (int, optional): Number of hidden layers for node
                processing. Defaults to 2.
            hidden_layers_edge (int, optional): Number of hidden layers for edge
                processing. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
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
        """Compute updates to the graph via successive message-passing blocks.

        Args:
            x (torch.Tensor): Input nodes.
            edge_index (torch.Tensor): Edge indices in COO format.
            edge_attr (torch.Tensor): Edge attributes.
            batch (int, optional): Batch size, forwarded to each block. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated nodes and edge attributes.
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
    """Overall graph processor: a stack of satellite message-passing Graph Net Blocks."""

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
        scatter: str="mean", disaggregated=False, attention=False, attention_mask=None, residuals: bool = False
    ):
        """Initialize the satellite graph processor.

        Args:
            mp_iterations (int, optional): Number of message-passing iterations
                (graph processor blocks). Defaults to 15.
            in_dim_node (int, optional): Input node feature dimension. Defaults to 128.
            in_dim_edge (int, optional): Input edge feature dimension. Defaults to 128.
            hidden_dim_node (int, optional): Number of nodes in hidden layers for
                node processing. Defaults to 128.
            hidden_dim_edge (int, optional): Number of nodes in hidden layers for
                edge processing. Defaults to 128.
            hidden_layers_node (int, optional): Number of hidden layers for node
                processing. Defaults to 2.
            hidden_layers_edge (int, optional): Number of hidden layers for edge
                processing. Defaults to 2.
            norm_type (str, optional): Normalization type, one of "LayerNorm",
                "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm", or None.
                Defaults to "LayerNorm".
            dropout (float, optional): Dropout probability. Defaults to 0.
            scatter (str, optional): Aggregation to use for incoming edge messages in
                the (non-disaggregated, non-attention) node model, "sum" or "mean".
                Defaults to "mean".
            disaggregated (bool, optional): If True, uses
                ``NodeSatelliteProcessorDisaggregated`` for the node model in each
                block. Defaults to False.
            attention (bool, optional): If True (and ``disaggregated`` is False), uses
                ``NodeSatelliteProcessorAttention`` for the node model in each block.
                Requires ``attention_mask``. Defaults to False.
            attention_mask (torch.Tensor, optional): Attention mask forwarded to each
                block when ``attention=True``. Defaults to None.
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
                    norm_type, dropout=dropout, scatter=scatter,disaggregated=disaggregated, attention=attention, attention_mask=attention_mask, residuals=residuals
                )
            )

        #print("set up, satellite version")
        #print("after loop", self.batched_data)
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute updates to the graph via successive message-passing blocks.

        Args:
            x (torch.Tensor): Input nodes.
            edge_index (torch.Tensor): Edge indices in COO format.
            edge_attr (torch.Tensor): Edge attributes.
            batch (int, optional): Batch size, forwarded to each block. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Updated nodes and edge attributes.
        """
        #print("graph block", batch)
        for block in self.blocks:
            #print("in graph processor")
            #print(x.size(), edge_index.size(), edge_attr.size())
            #print("in block")
            #print(edge_index[0])
            x, edge_attr, _ = block(x, edge_index, edge_attr, batch)#, batch=torch.tensor(np.arange(x.size()[0])))

        return x, edge_attr




