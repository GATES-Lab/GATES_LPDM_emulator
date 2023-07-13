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

from graphnet_LPDM_emulator.model.layers.graph_net_block import MLP

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
        final_activation=None, concat_neighbours=False
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
        NEEDS UPDATING!

        """

        super().__init__()
        self.residuals = residuals
        self.use_checkpointing = use_checkpointing
        self.num_latlons = len(lat_lons)
        self.concat_neighbours = concat_neighbours
        self.n_neighbours = n_neighbours
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
            if self.concat_neighbours:
                # previous way models where trained with - only link top 3 physically close neighbours
                # see except clause before
                linked_neighbours = 0 
                n=0
                while linked_neighbours < n_neighbours:
                    try:
                        distance = knn_distances[ordered_knn_distances[n]]
                        h = h_points[ordered_knn_distances[n]]
                        loc_neighbour = h3.h3_to_geo(h)
                        edge_sources.append(self.base_h3_map[h])
                        edge_targets.append(idx)
                        self.h3_distances.append([distance])
                        linked_neighbours += 1 
                        n+=1

                    except KeyError:
                        # this except will be triggered if h (one of the neighbouring points to h3_index) is not in the list
                        # this can only happen if using a reduced domain (whole_world = False), at the edges of this domain
                        n+=1
                        #continue    
                                  
            else:
                # previous way models where trained with - only link top 3 physically close neighbours
                # see except clause before
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

        if self.concat_neighbours:
            self.node_decoder = MLP(
            self.n_neighbours*input_dim,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
        )          # no normalising here?
            
            print("MLP shape", 3*input_dim, hidden_dim_decoder)

        else:
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
        
        #print("after rearrange 1", processor_features.size(), processor_features.dtype, torch.from_numpy(self.edge_weights).size())
        #print("for mult", processor_features[:,:, self.edge_index[0,:]].size())
        processor_features = einops.rearrange(processor_features, "(b n) f -> b f n", b=batch_size)
        
        if self.concat_neighbours:
            print("concatting neighbours")
            
            print(processor_features.size(), self.edge_index)

            ## loop is clunky! maybe einops has a solution
            print(processor_features[:,:, self.edge_index[0,:]].size())
            print(processor_features[:,:, self.edge_index[0,::self.n_neighbours]].size(), processor_features[:,:, self.edge_index[0,1::self.n_neighbours]].size())
            # this could go wrong 
            scattered_processor_features = torch.cat([processor_features[..., self.edge_index[0,n::self.n_neighbours]] for n in range(self.n_neighbours)], dim=-2)
            print(scattered_processor_features.size())

            #scattered_processor_features = einops.rearrange(scattered_processor_features, "b f n -> (b n) f", b=batch_size)
            processor_features = scattered_processor_features

            print(processor_features.size())
            
        else:
            processor_features = torch.multiply(processor_features[:,:, self.edge_index[0,:]], torch.flatten(self.edge_weights))
            #print("doing scatter", processor_features.dtype, torch.from_numpy(self.edge_weights.astype(np.float32)).dtype, self.edge_weights.astype(float).dtype, self.edge_weights.dtype)
            processor_features = scatter_mean(src=processor_features, index=self.edge_index[1,:]) 

            #processor_features_by_node = np.zeros((processor_features.size()[0], #start_features.size()[1], processor_features.size()[-1]))
            #for n in np.unique(self.edge_index[1,:]): 
            #    processor_features_by_node = np.where(self.edge_index[1,:]==n)

        processor_features = einops.rearrange(processor_features, "b f n -> (b n) f", b=batch_size)
        #print("after scatter", processor_features.size(), processor_features.dtype)


        out = self.node_decoder(processor_features)  # Decode to output dim from hidden size
            #print("done it")
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)
        
        
        return out