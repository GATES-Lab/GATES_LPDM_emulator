"""Decoders to decode from the Processor graph to the original graph with updated values

In the original paper the decoder is described as

The Decoder maps back to physical data defined on a latitude/longitude grid. The underlying graph is
again bipartite, this time mapping icosahedron→lat/lon.
The inputs to the Decoder come from the Processor, plus a skip connection back to the original
state of the 78 atmospheric variables onthe latitude/longitude grid.
The output of the Decoder is the predicted 6-hour change in the 78 atmospheric variables,
which is then added to the initial state to produce the new state. We found 6 hours to be a good
balance between shorter time steps (simpler dynamics to model but more iterations required during
rollout) and longer time steps (fewer iterations required during rollout but modeling
more complex dynamics)

"""
import torch
import einops
import h3
import numpy as np
from torch_geometric.data import Data
from torch_scatter import scatter_mean

from graph_weather.models.layers.graph_net_block import MLP, GraphProcessor

class Decoder(torch.nn.Module):
    """Decoder graph module"""

    def __init__(
        self,
        lat_lons,
        whole_world = False,
        resolution: int = 2,
        input_dim: int = 256,
        output_dim: int = 78,
        output_edge_dim: int = 256,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        mlp_norm_type: str = "LayerNorm",
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        residuals: bool = False,
        use_checkpointing: bool = False,
        dropout=0,
    ):
        """
        Decoder from latent graph to lat/lon graph

        Args:
            lat_lons: List of (lat,lon) points
            resolution: H3 resolution level
            input_dim: Input node dimension
            output_dim: Output node dimension
            output_edge_dim: Edge dimension
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Whether to use gradient checkpointing or not


         modifications to og code:
            - adapted to work in the whole world or only for the area defined by the lat lon coords 
            - og code connects each latlon point to the closest mesh point and all of its neighbours. changed so it's only connected to the three closest of these
        to add
            - 
 
        """

        super().__init__()
        self.residuals = residuals
        self.use_checkpointing = use_checkpointing
        self.num_latlons = len(lat_lons)

        if whole_world:
            self.base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
            self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        
        else:
            self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]   
            self.base_h3_grid = sorted(list(set(self.h3_grid)))     

        self.num_h3 = len(self.base_h3_grid)

        self.h3_to_index = {}
        h_index = len(self.base_h3_grid)
        for h in self.base_h3_grid:
            if h not in self.h3_to_index:
                h_index -= 1
                self.h3_to_index[h] = h_index

        self.h3_mapping = {}
        for h, value in enumerate(self.h3_grid):
            self.h3_mapping[h + self.num_h3] = value

        # Build the default graph
        ## changed here
        nodes = torch.ones(
            (len(lat_lons) + self.num_h3, input_dim), dtype=torch.float
        )
        # Extra starting ones for appending to inputs, could 'learn' good starting points
        ## changed here
        self.latlon_nodes = torch.ones((len(lat_lons), input_dim), dtype=torch.float)
        # Get connections between lat nodes and h3 nodes TODO Paper makes it seem like the 3
        #  closest iso points map to the lat/lon point Do kring 1 around current h3 cell,
        #  and calculate distance between all those points and the lat/lon one, choosing the
        #  nearest N (3) For a bit simpler, just include them all with their distances
        edge_sources = []
        edge_targets = []
        self.h3_to_lat_distances = []
        """ connecting each mesh node and its neighbours with closest grid nodes
        for node_index, h_node in enumerate(self.h3_grid):
            # Get h3 index
            h_points = h3.k_ring(h_node, 1)
            for h in h_points:
                distance = h3.point_dist(lat_lons[node_index], h3.h3_to_geo(h), unit="rads")
                try:
                    self.h3_to_lat_distances.append([np.sin(distance), np.cos(distance)])
                    edge_sources.append(self.h3_to_index[h])
                    edge_targets.append(node_index + self.num_h3)
                except KeyError:
                    # this except will be triggered if h (one of the neighbouring points to h3_index) is not in the list
                    # this can only happen if using a reduced domain (whole_world = False), at the edges of this domain
                    continue
        """
        # connecting each latlon node to its closest three mesh nodes
        for node_index, h_node in enumerate(self.h3_grid):
            # Get h3 index
            h_points = list(h3.k_ring(h_node, 1))
            knn_distances = [h3.point_dist(lat_lons[node_index], h3.h3_to_geo(h), unit="rads") for h in h_points]
            ordered_knn_distances = np.argsort(knn_distances)
            for n in range(3): # ie the three neighbours closest to h_node
                try:
                    distance = knn_distances[ordered_knn_distances[n]]
                    h = h_points[ordered_knn_distances[n]]
                    edge_sources.append(self.h3_to_index[h])
                    edge_targets.append(node_index + self.num_h3)
                    self.h3_to_lat_distances.append([np.sin(distance), np.cos(distance)])

                except KeyError:
                    # this except will be triggered if h (one of the neighbouring points to h3_index) is not in the list
                    # this can only happen if using a reduced domain (whole_world = False), at the edges of this domain
                    continue    





        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        self.h3_to_lat_distances = torch.tensor(self.h3_to_lat_distances, dtype=torch.float)

        # Use normal graph as its a bit simpler
        self.graph = Data(x=nodes, edge_index=edge_index, edge_attr=self.h3_to_lat_distances)
        #print("decoder: setting up edge dec")
        self.edge_encoder = MLP(
            2, output_edge_dim, hidden_dim_processor_edge, 2, mlp_norm_type, self.use_checkpointing, dropout=dropout
        )
        #print("decoder: setting up graph processor")
        self.graph_processor = GraphProcessor(
            mp_iterations=1,
            in_dim_node=input_dim,
            in_dim_edge=output_edge_dim,
            hidden_dim_node=hidden_dim_processor_node,
            hidden_dim_edge=hidden_dim_processor_edge,
            hidden_layers_node=hidden_layers_processor_node,
            hidden_layers_edge=hidden_layers_processor_edge,
            norm_type=mlp_norm_type, dropout=dropout
        )
        #print("decoder: setting up node dec")
        self.node_decoder = MLP(
            input_dim,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout
        )

    def forward(
        self, processor_features: torch.Tensor, start_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            processor_features: Processed features in shape [B*Nodes, Features]
            start_features: Original input features to the encoder, with shape [B, Nodes, Features]

        Returns:
            Updated features for model
        """
        #print("in decoder forward")
        batch_size = start_features.shape[0]
        self.graph = self.graph.to(processor_features.device)
        #print("decoder: encoding edges")
        edge_attr = self.edge_encoder(self.graph.edge_attr)  # Update attributes based on distance
        edge_attr = einops.repeat(edge_attr, "e f -> (repeat e) f", repeat=batch_size)

        edge_index = torch.cat(
            [
                self.graph.edge_index + i * torch.max(self.graph.edge_index) + i
                for i in range(batch_size)
            ],
            dim=1,
        )

        # Readd nodes to match graph node number
        self.latlon_nodes = self.latlon_nodes.to(processor_features.device)
        features = einops.rearrange(processor_features, "(b n) f -> b n f", b=batch_size)
        features = torch.cat(
            [features, einops.repeat(self.latlon_nodes, "n f -> b n f", b=batch_size)], dim=1
        )
        features = einops.rearrange(features, "b n f -> (b n) f")
        #print("decoder: doing graph processing")
        #print(features.size(), edge_index.size(), edge_attr.size())
        out, _ = self.graph_processor(features, edge_index, edge_attr)  # Message Passing
        # Remove the h3 nodes now, only want the latlon ones
        #print("decoder: decoding nodes")
        out = self.node_decoder(out)  # Decode to output dim from hidden size
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)
        _, out = torch.split(out, [self.num_h3, self.num_latlons], dim=1)


        if self.residuals:
            out = out + start_features  # residual connection
            #print("doing residuals")
        
        #out = torch.squeeze(out)
        
        return out

class SatelliteDecoder(torch.nn.Module):
    """Decoder graph module"""

    def __init__(
        self,
        lat_lons,
        h_grid = None,
        whole_world = False,
        resolution: int = 2,
        input_dim: int = 256,
        output_dim: int = 78,
        mlp_norm_type: str = "LayerNorm",
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        residuals: bool = False,
        use_checkpointing: bool = False,
        dropout=0,
        n_neighbours=3,
        final_activation=None,
    ):
        """
        Decoder from latent graph to lat/lon graph

        Args:
            lat_lons: List of (lat,lon) points
            resolution: H3 resolution level
            input_dim: Input node dimension
            output_dim: Output node dimension
            output_edge_dim: Edge dimension
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Whether to use gradient checkpointing or not


         modifications to og code:
            - adapted to work in the whole world or only for the area defined by the lat lon coords 
            - og code connects each latlon point to the closest mesh point and all of its neighbours. changed so it's only connected to the three closest of these
        to add
            - 
 
        """

        super().__init__()
        self.residuals = residuals
        self.use_checkpointing = use_checkpointing
        self.num_latlons = len(lat_lons)
        #print("in satellite decoder")
        if h_grid is None:
            if whole_world:
                self.base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
                self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
            
            else:
                # h3_grid - list of all mesh node IDs, corresponding to the lat-lons
                self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]   
                # base_h3_grid - sorted set of all mesh node IDs
                self.base_h3_grid = sorted(list(set(self.h3_grid)))     

        else:
            self.h3_grid = h_grid
            self.base_h3_grid = sorted(list(set(self.h3_grid)))  


        self.num_h3 = len(self.base_h3_grid)
        self.base_h3_map = {h_i: i for i, h_i in enumerate(self.base_h3_grid)}



        # Extra starting ones for appending to inputs, could 'learn' good starting points
        ## changed here
        
        # Get connections between lat nodes and h3 nodes TODO Paper makes it seem like the 3
        #  closest iso points map to the lat/lon point Do kring 1 around current h3 cell,
        #  and calculate distance between all those points and the lat/lon one, choosing the
        #  nearest N (3) For a bit simpler, just include them all with their distances
        edge_sources = []
        edge_targets = []
        self.h3_distances = []
        # connecting each latlon node to its closest three mesh nodes
        for idx, h3_point in enumerate(self.h3_grid):
            # Get h3 index
            lat_lon = lat_lons[idx]
            # get the list of mesh nodes closest to the corresponding mesh hode to the selected latlon node
            h_points = list(h3.k_ring(h3_point, 1))
            knn_distances = [h3.point_dist(lat_lon, h3.h3_to_geo(h), unit="km") for h in h_points]

            ordered_knn_distances = np.argsort(knn_distances)
            for n in range(n_neighbours): # ie the n neighbours closest to h_node
                try:
                    distance = knn_distances[ordered_knn_distances[n]]
                    h = h_points[ordered_knn_distances[n]]
                    loc_neighbour = h3.h3_to_geo(h)
                    edge_sources.append(self.base_h3_map[h])
                    edge_targets.append(idx)
                    self.h3_distances.append([distance])

                except KeyError:
                    # this except will be triggered if h (one of the neighbouring points to h3_index) is not in the list
                    # this can only happen if using a reduced domain (whole_world = False), at the edges of this domain
                    continue    

        self.edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        self.h3_distances = np.array(self.h3_distances)
        
        self.edge_weights = np.zeros_like(self.h3_distances)

        for n in np.unique(edge_targets):
            same_edges = np.where(edge_targets==n)

            self.edge_weights[same_edges] = len(same_edges[0])*(1/self.h3_distances[same_edges])/np.sum([1/i for i in self.h3_distances[same_edges]])
        
        self.edge_weights = torch.tensor(self.edge_weights, dtype=torch.float32)

        self.node_decoder = MLP(
        input_dim,
        output_dim,
        hidden_dim_decoder,
        hidden_layers_decoder,
        None,
        self.use_checkpointing, dropout=dropout, final_activation=final_activation
    )          # no normalising here?

    def forward(
        self, processor_features: torch.Tensor, start_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Adds features to the encoding graph

        Args:
            processor_features: Processed features in shape [B*Nodes, Features]
            start_features: Original input features to the encoder, with shape [B, Nodes, Features]

        Returns:
            Updated features for model
        """
        #print("in satellite decoder forward")
        batch_size = start_features.shape[0]

        self.edge_weights = self.edge_weights.to(start_features.device)
        self.edge_index = self.edge_index.to(start_features.device)

        #print(processor_features.size(), processor_features.dtype, self.edge_index.size(), start_features.size())
        processor_features = einops.rearrange(processor_features, "(b n) f -> b f n", b=batch_size)
        #print("after rearrange 1", processor_features.size(), processor_features.dtype, torch.from_numpy(self.edge_weights).size())
        #print("for mult", processor_features[:,:, self.edge_index[0,:]].size())
        processor_features = torch.multiply(processor_features[:,:, self.edge_index[0,:]], torch.flatten(self.edge_weights))
        #print("doing scatter", processor_features.dtype, torch.from_numpy(self.edge_weights.astype(np.float32)).dtype, self.edge_weights.astype(float).dtype, self.edge_weights.dtype)
        processor_features = scatter_mean(src=processor_features, index=self.edge_index[1,:]) 

        #processor_features_by_node = np.zeros((processor_features.size()[0], #start_features.size()[1], processor_features.size()[-1]))
        #for n in np.unique(self.edge_index[1,:]): 
        #    processor_features_by_node = np.where(self.edge_index[1,:]==n)



        #print("after scatter", processor_features.size(), processor_features.dtype)
        processor_features = einops.rearrange(processor_features, "b f n -> (b n) f", b=batch_size)
        start_features = einops.rearrange(start_features, "b n f -> (b n) f", b=batch_size)
        #print("after rearrange", processor_features.size(), processor_features.dtype)
        #print("rearranged")
        #print(self.residuals)
        if self.residuals:
            #print(torch.mean(processor_features), torch.max(processor_features), torch.min(processor_features))
            #print(torch.mean(start_features), torch.max(start_features), torch.min(start_features))
            processor_features = torch.cat([processor_features, start_features], dim=1)
            print("added residuals", processor_features.size())
        
        #print(processor_features.size(), processor_features.dtype)


        out = self.node_decoder(processor_features)  # Decode to output dim from hidden size
        #print("done it")
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)
        
        
        return out