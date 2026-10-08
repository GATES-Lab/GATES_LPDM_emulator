"""Decoders to decode from the Processor graph to the original graph with updated values

author: Elena Fillola @elenafillo

"""
import torch
import einops
import h3
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean

from model.layers.graph_net_block import MLP
import math



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
        final_activation=None, concat_neighbours=False, idx_latlon=None, append_latlon=False, concat_neighbours_2=False,
        skip_dim: int = 0, skip_mode: str = "encoded",
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
            skip_dim: Grid-level skip connection. 0 (default) = off: every lat/lon cell is decoded
                from the blend of its nearest mesh nodes alone. > 0: the first ``skip_dim`` input
                features of the cell itself (``start_features[..., :skip_dim]``) are joined to that
                blend before the decoder MLP, so the decoder sees the cell at the resolution of the
                lat/lon grid (the mesh is coarser than the grid).
            skip_mode: "encoded" (default) passes the cell's features through an MLP of the
                decoder's width first (``skip_dim -> input_dim``, with the decoder's norm);
                "inputs" joins them as they are.



        """

        super().__init__()
        self.residuals = residuals
        self.use_checkpointing = use_checkpointing
        self.num_latlons = len(lat_lons)
        self.concat_neighbours = concat_neighbours
        self.concat_neighbours_2 = concat_neighbours_2
        self.n_neighbours = n_neighbours
        self.append_latlon = append_latlon

        self.skip_dim = int(skip_dim)
        self.skip_mode = skip_mode
        self.skip_encoder = None
        if self.skip_dim < 0:
            raise ValueError(f"skip_dim must be >= 0, got {skip_dim}")
        if self.skip_dim:
            if skip_mode not in ("encoded", "inputs"):
                raise ValueError(f"skip_mode must be 'encoded' or 'inputs', got {skip_mode!r}")
            if concat_neighbours or concat_neighbours_2:
                raise ValueError("the grid-level skip (skip_dim > 0) is implemented for the default "
                                 "decoder only, not with concat_neighbours / concat_neighbours_2")

        if append_latlon:
            assert idx_latlon is not None, "pass idx latlon!"
            m = np.max(np.abs(idx_latlon))
            self.normalised_idx_latlon = torch.tensor(np.array([(a[0]/m, a[1]/m) for a in idx_latlon]), dtype=torch.float32)

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
                    except IndexError:
                        print(linked_neighbours)
                        print(idx, h3_point)
                        print(lat_lon)
                        print(h_points)
                        print(knn_distances)
                    
                        exit()
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
        self.norm_distances = torch.tensor(self.h3_distances/50, dtype=torch.float32) ### note that 50 here is because of resolutioN!!!!! 

        self.edge_weights = np.zeros_like(self.h3_distances)

        for n in np.unique(edge_targets):
            same_edges = np.where(edge_targets==n)

            self.edge_weights[same_edges] = len(same_edges[0])*(1/self.h3_distances[same_edges])/np.sum([1/i for i in self.h3_distances[same_edges]])
        
        self.edge_weights = torch.tensor(self.edge_weights, dtype=torch.float32)

        if self.concat_neighbours:
            self.node_decoder = MLP(
            self.n_neighbours*(1+input_dim) + 2*self.append_latlon,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
        )          # no normalising here?

        if self.concat_neighbours_2:
            """
            self.pre_decoder = MLP(
            1+input_dim + 2*self.append_latlon,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
            )          # no normalising here?"""
            self.node_decoder = MLP(
            self.n_neighbours*(1+input_dim),
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
            )          # no normalising here?
                       
            print("MLP shape", 3*input_dim, hidden_dim_decoder)

        else:
            # grid-level skip: width of what is joined to the blended mesh features (0 = off)
            skip_width = 0
            if self.skip_dim:
                skip_width = input_dim if self.skip_mode == "encoded" else self.skip_dim
            self.node_decoder = MLP(
            input_dim + 2*self.append_latlon + skip_width,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
        )          # no normalising here?
            if self.skip_dim and self.skip_mode == "encoded":
                # built AFTER the decoder MLP, and only when the skip is on, so a model without
                # the skip draws the same initial weights as before the option existed
                self.skip_encoder = MLP(
                    self.skip_dim,
                    input_dim,
                    hidden_dim_decoder,
                    hidden_layers_decoder,
                    mlp_norm_type,
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
        #print("in satellite decoder forward")
        batch_size = start_features.shape[0]

        self.edge_weights = self.edge_weights.to(start_features.device)
        self.edge_index = self.edge_index.to(start_features.device)
        self.norm_distances = self.norm_distances.to(start_features.device)


        processor_features = einops.rearrange(processor_features, "(b n) f -> b f n", b=batch_size)
        
        if self.concat_neighbours:
            print("warning, this has not been tested recently")
            scattered_processor_features = torch.cat([processor_features[..., self.edge_index[0,n::self.n_neighbours]] for n in range(self.n_neighbours)], dim=-2)
            #print(scattered_processor_features.size())


            processor_features = scattered_processor_features


        elif self.concat_neighbours_2:
            print("warning, this has not been tested recently")
            catting = torch.cat([processor_features[..., self.edge_index[0,:]], einops.repeat(self.norm_distances, "l f -> n f l", n=batch_size)], dim=1)


            processor_features = concat_group_by(catting, self.edge_index[1,:].squeeze())


            #print(processor_features.size(),processor_features.dtype )

        else:
            processor_features = torch.multiply(processor_features[:,:, self.edge_index[0,:]], torch.flatten(self.edge_weights))

            processor_features = scatter_mean(src=processor_features, index=self.edge_index[1,:]) 


        if self.append_latlon:
            normalised_idx_latlon = einops.repeat(self.normalised_idx_latlon, "e f -> n f e", n=batch_size)
            normalised_idx_latlon = normalised_idx_latlon.to(start_features.device)

            processor_features = torch.cat([processor_features, normalised_idx_latlon], dim=1)


        processor_features = einops.rearrange(processor_features, "b f n -> (b n) f", b=batch_size)

        if self.skip_dim:
            # grid-level skip: the cell's own input features, next to the blend of its mesh nodes
            skip = einops.rearrange(start_features[..., :self.skip_dim], "b n f -> (b n) f")
            if self.skip_encoder is not None:
                skip = self.skip_encoder(skip)
            processor_features = torch.cat([processor_features, skip], dim=-1)


        out = self.node_decoder(processor_features)
        # Decode to output dim from hidden size
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)


        return out


class SatelliteDecoderPredictor(torch.nn.Module):
    """Decoder graph module"""

    def __init__(
        self,
        lat_lons,
        h_grid = None,
        whole_world = False,
        resolution: int = 2,
        input_dim: int = 256,
        output_dim: int = 78,
        num_classes:int = 4,
        mlp_norm_type: str = "LayerNorm",
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        residuals: bool = False,
        use_checkpointing: bool = False,
        dropout=0,
        n_neighbours=3,
        final_activation=None, concat_neighbours=False, idx_latlon=None, append_latlon=False, concat_neighbours_2=False
    ):
        """
        Decoder from latent graph to an vector of dimension (num_classes,). This is an extension to SatelliteDecoder, which produces a per-node output for the whole grid, followed by a linear layer to produce num_classes outputs.

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

        This pulls each mesh node's processed embedding into the lat/lon nodes it feeds, does an inverse-distance-weighted 
        average (scatter_mean over the weighted contributions).
        Following this an MLP applied to every lat/lon node independently, mapping its aggregated feature → output_dim.
        Flatten all N_latlon node vectors into one long vector per sample → [B, N_latlon·output_dim].
        linear_class = Linear(num_latlons·output_dim → num_classes) — a single fully-connected layer over the whole flattened grid produces the num_classes outputs.

        """

        super().__init__()
        self.residuals = residuals
        self.use_checkpointing = use_checkpointing
        self.num_latlons = len(lat_lons)
        self.concat_neighbours = concat_neighbours
        self.concat_neighbours_2 = concat_neighbours_2
        self.n_neighbours = n_neighbours
        self.append_latlon = append_latlon

        if append_latlon:
            assert idx_latlon is not None, "pass idx latlon!"
            m = np.max(np.abs(idx_latlon))
            self.normalised_idx_latlon = torch.tensor(np.array([(a[0]/m, a[1]/m) for a in idx_latlon]), dtype=torch.float32)


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
                    except IndexError:
                        print(linked_neighbours)
                        print(idx, h3_point)
                        print(lat_lon)
                        print(h_points)
                        print(knn_distances)
                    
                        exit()
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
        self.norm_distances = torch.tensor(self.h3_distances/50, dtype=torch.float32) ### note that 50 here is because of resolutioN!!!!! 

        self.edge_weights = np.zeros_like(self.h3_distances)

        for n in np.unique(edge_targets):
            same_edges = np.where(edge_targets==n)

            self.edge_weights[same_edges] = len(same_edges[0])*(1/self.h3_distances[same_edges])/np.sum([1/i for i in self.h3_distances[same_edges]])
        
        self.edge_weights = torch.tensor(self.edge_weights, dtype=torch.float32)

        if self.concat_neighbours:
            self.node_decoder = MLP(
            self.n_neighbours*(1+input_dim) + 2*self.append_latlon,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
        )          # no normalising here?

        if self.concat_neighbours_2:
            """
            self.pre_decoder = MLP(
            1+input_dim + 2*self.append_latlon,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
            )          # no normalising here?"""
            self.node_decoder = MLP(
            self.n_neighbours*(1+input_dim),
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
            )          # no normalising here?
                       
            print("MLP shape", 3*input_dim, hidden_dim_decoder)

        else:
            self.node_decoder = MLP(
            input_dim + 2*self.append_latlon,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
        )          # no normalising here?
        
        self.linear_class = torch.nn.Linear(self.num_latlons*output_dim, num_classes)

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

        batch_size = start_features.shape[0]

        self.edge_weights = self.edge_weights.to(start_features.device)
        self.edge_index = self.edge_index.to(start_features.device)
        self.norm_distances = self.norm_distances.to(start_features.device)


        processor_features = einops.rearrange(processor_features, "(b n) f -> b f n", b=batch_size)
        
        if self.concat_neighbours:
            print("warning, this has not been tested recently")
            scattered_processor_features = torch.cat([processor_features[..., self.edge_index[0,n::self.n_neighbours]] for n in range(self.n_neighbours)], dim=-2)

            processor_features = scattered_processor_features


        elif self.concat_neighbours_2:
            print("warning, this has not been tested recently")

            catting = torch.cat([processor_features[..., self.edge_index[0,:]], einops.repeat(self.norm_distances, "l f -> n f l", n=batch_size)], dim=1)


            processor_features = concat_group_by(catting, self.edge_index[1,:].squeeze())



        else:
            processor_features = torch.multiply(processor_features[:,:, self.edge_index[0,:]], torch.flatten(self.edge_weights))
 
            processor_features = scatter_mean(src=processor_features, index=self.edge_index[1,:]) 


        if self.append_latlon:
            normalised_idx_latlon = einops.repeat(self.normalised_idx_latlon, "e f -> n f e", n=batch_size)
            normalised_idx_latlon = normalised_idx_latlon.to(start_features.device)

            processor_features = torch.cat([processor_features, normalised_idx_latlon], dim=1)

        processor_features = einops.rearrange(processor_features, "b f n -> (b n) f", b=batch_size)

    

        out = self.node_decoder(processor_features)  
        '''
        out = einops.rearrange(out, "(b n) f -> b n f", b=batch_size)
        '''
        out = einops.rearrange(out, "(b n) f -> b (n f)", b=batch_size)

        out = self.linear_class(out)
        return out
    





class SatelliteDecoderConvPredictor(torch.nn.Module):
    """ 
    Decoder from latent graph to an vector of dimension (num_classes,). This is an extension to SatelliteDecoder, which produces a per-node output for the whole grid, followed by a conv block and a linear layer to produce num_classes outputs.
    """

    def __init__(
        self,
        lat_lons,
        h_grid = None,
        whole_world = False,
        resolution: int = 2,
        input_dim: int = 256,
        output_dim: int = 78,
        num_classes:int = 4,
        mlp_norm_type: str = "LayerNorm",
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        residuals: bool = False,
        use_checkpointing: bool = False,
        dropout=0,
        n_neighbours=3,
        final_activation=None, concat_neighbours=False, idx_latlon=None, append_latlon=False, concat_neighbours_2=False,input_height=100, input_width=100
    ):
        """
        Decoder from latent graph to lat/lon graph
        This pulls each mesh node's processed embedding into the lat/lon nodes it feeds, does an inverse‑distance‑weighted 
        average (scatter_mean over the weighted contributions), and reshapes the N_latlon = H·W nodes into an image of F = node_dim channels over the size × size grid. 
        So after this step the latent graph has been turned back into a spatial feature map.
        This is followed by the a CNN block and a linear head.
        Two stride‑2 convs downsample the spatial map by 4× and compress channels (node_dim→32→8), 
        then a flatten + Linear produce the final [B, num_classes] output. num_classes is 1 
        (summed background) or 4 (per‑boundary N/S/E/W)

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

       """

        super().__init__()
        self.residuals = residuals
        self.use_checkpointing = use_checkpointing
        self.num_latlons = len(lat_lons)
        self.concat_neighbours = concat_neighbours
        self.concat_neighbours_2 = concat_neighbours_2
        self.n_neighbours = n_neighbours
        self.append_latlon = append_latlon

        if append_latlon:
            assert idx_latlon is not None, "pass idx latlon!"
            m = np.max(np.abs(idx_latlon))
            self.normalised_idx_latlon = torch.tensor(np.array([(a[0]/m, a[1]/m) for a in idx_latlon]), dtype=torch.float32)


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
                    except IndexError:
                        print(linked_neighbours)
                        print(idx, h3_point)
                        print(lat_lon)
                        print(h_points)
                        print(knn_distances)
                    
                        exit()
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
        self.norm_distances = torch.tensor(self.h3_distances/50, dtype=torch.float32) ### note that 50 here is because of resolutioN!!!!! 

        self.edge_weights = np.zeros_like(self.h3_distances)

        for n in np.unique(edge_targets):
            same_edges = np.where(edge_targets==n)

            self.edge_weights[same_edges] = len(same_edges[0])*(1/self.h3_distances[same_edges])/np.sum([1/i for i in self.h3_distances[same_edges]])
        
        self.edge_weights = torch.tensor(self.edge_weights, dtype=torch.float32)

        if self.concat_neighbours:
            self.node_decoder = MLP(
            self.n_neighbours*(1+input_dim) + 2*self.append_latlon,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
        )          # no normalising here?

        if self.concat_neighbours_2:
            """
            self.pre_decoder = MLP(
            1+input_dim + 2*self.append_latlon,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
            )          # no normalising here?"""
            self.node_decoder = MLP(
            self.n_neighbours*(1+input_dim),
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
            )          # no normalising here?
                       
            print("MLP shape", 3*input_dim, hidden_dim_decoder)

        else:
            self.node_decoder = MLP(
            input_dim + 2*self.append_latlon,
            output_dim,
            hidden_dim_decoder,
            hidden_layers_decoder,
            None,
            self.use_checkpointing, dropout=dropout, final_activation=final_activation
        )          # no normalising here?
            

        # Conv block input channels MUST match the processor feature dim (input_dim),
        # since forward() reshapes processor_features into [B, input_dim, H, W].
        conv_hidden_channels = 32
        conv_out_channels = 8
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(input_dim, conv_hidden_channels, kernel_size=3, stride=2, padding=1),  # /2
            torch.nn.ReLU(),
            torch.nn.Conv2d(conv_hidden_channels, conv_out_channels, kernel_size=3, stride=2, padding=1),  # /4
            torch.nn.ReLU(),
        )

        # Two stride-2 conv layers → spatial size reduced by 2**2 = 4
        num_stride2_convs = 2
        conv_reduction = 2 ** num_stride2_convs
        assert input_height >= 1 and input_width >= 1, (
            f"input_height and input_width must be >= 1, got "
            f"input_height={input_height}, input_width={input_width}"
        )
        out_H = math.ceil(input_height / conv_reduction)
        out_W = math.ceil(input_width / conv_reduction)
        flattened_size = conv_out_channels * out_H * out_W

        self.input_height = input_height
        self.input_width = input_width
        print('input height',self.input_height)
        print('input width',self.input_width)

        self.linear_class = torch.nn.Linear(flattened_size, num_classes)

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

        batch_size = start_features.shape[0]

        self.edge_weights = self.edge_weights.to(start_features.device)
        self.edge_index = self.edge_index.to(start_features.device)
        self.norm_distances = self.norm_distances.to(start_features.device)


        processor_features = einops.rearrange(processor_features, "(b n) f -> b f n", b=batch_size)
        
        
        processor_features = torch.multiply(processor_features[:,:, self.edge_index[0,:]], torch.flatten(self.edge_weights))

        processor_features = scatter_mean(src=processor_features, index=self.edge_index[1,:]) 


        processor_features = einops.rearrange(processor_features, "b f (h w) -> b f h w", b=batch_size, h=self.input_height, w=self.input_width)


        out = self.conv_block(processor_features)

        out = out.view(batch_size,-1)
        

        out = self.linear_class(out)

        return out

def concat_group_by(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    # Step 1: Count the occurrences of each unique value in the index tensor.
    index_count = torch.bincount(index)
    
    # Step 2: Calculate how many padding (zero-fill) elements are needed for each group.
    fill_count = index_count.max() - index_count
    
    # Step 3: Create a tensor of zeros for padding. 
    # Since we are padding along the 'nodes' dimension, we create zero padding for each feature and batch.
    fill_zeros = torch.zeros(x.shape[0], x.shape[1], fill_count.sum(), device=x.device)
    
    # Step 4: Create a tensor of indices that corresponds to the padding (zero-fill) elements.
    fill_index = torch.arange(0, fill_count.shape[0], device=x.device).repeat_interleave(fill_count)
    
    # Step 5: Concatenate the original `index` tensor with the `fill_index` tensor.
    # This expands the `index` to account for the padded elements.
    index_ = torch.cat([index, fill_index], dim=0)
    
    # Step 6: Concatenate the original `x` tensor with the `fill_zeros` tensor along the `nodes` dimension.
    x_ = torch.cat([x, fill_zeros], dim=2)
    
    # Step 7: Sort the concatenated tensor based on the sorted `index_` to group the elements.
    # Sorting happens across the `nodes` dimension.
    sort_indices = torch.argsort(index_, stable=True)
    x_ = x_[:, :, sort_indices]
    
    # Step 8: Reshape the tensor into the desired shape.
    # The output shape will be [n_batch, n_features, number of groups, max group size].
    x_ = x_.view(x.shape[0], x.shape[1], index_count.shape[0], index_count.max())

    # Step 9: Flatten the `n_features` and `max_group_size` dimensions.
    x_ = x_.permute(0, 2, 1, 3)  # Rearrange to [n_batch, number_of_groups, n_features, max_group_size].
    x_ = x_.reshape(x_.shape[0], x_.shape[1], -1)  # Flatten n_features * max_group_size.
    
    # Step 10: Permute the tensor to get the final shape [n_batch, n_features * max_group_size, number_of_groups].
    x_ = x_.permute(0, 2, 1)
    
    return x_




class ExitCurtainHead(SatelliteDecoderConvPredictor):
    """Whole-window head that predicts the particles' EXIT DISTRIBUTION over the domain boundary
    (``gates/data/exit_curtains.py``): the processed mesh nodes are blended back onto the lat/lon
    grid, reshaped to an image, passed through the two stride-2 convolutions of the background head
    and a linear layer to ``n_outputs`` logits (one per pooled curtain bin plus the remainder bin). The
    trainer applies a softmax, so the output is a probability vector.

    This is the background head of the dual model (``SatelliteDecoderConvPredictor``) with a wide
    output instead of one number, so a model with this head differs from the dual model's background
    head in the TARGET only. Differences: the per-node MLP that ``SatelliteDecoderConvPredictor``
    builds but never uses in ``forward`` is removed (it would only add untrained parameters to the
    checkpoint), and ``hidden_dim > 0`` inserts a hidden layer between the flattened feature map and the
    logits (``flattened -> hidden_dim -> n_outputs``; 0 = one linear layer, as in the background head).
    """

    def __init__(self, lat_lons, h_grid=None, whole_world=False, resolution=2, input_dim=256, n_outputs=1,
                 mlp_norm_type="LayerNorm", hidden_dim_decoder=128, hidden_layers_decoder=2,
                 use_checkpointing=False, dropout=0, n_neighbours=3, input_height=100, input_width=100,
                 hidden_dim=0):
        if int(n_outputs) < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        if int(hidden_dim) < 0:
            raise ValueError(f"hidden_dim must be >= 0, got {hidden_dim}")
        super().__init__(
            lat_lons=lat_lons, h_grid=h_grid, whole_world=whole_world, resolution=resolution,
            input_dim=input_dim, output_dim=1, num_classes=int(n_outputs), mlp_norm_type=mlp_norm_type,
            hidden_dim_decoder=hidden_dim_decoder, hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing, dropout=dropout, n_neighbours=n_neighbours,
            input_height=input_height, input_width=input_width,
        )
        # the parent's per-node MLP is not part of its forward pass: drop the dead parameters
        del self.node_decoder
        self.n_outputs = int(n_outputs)
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim:
            flattened = self.linear_class.in_features
            self.linear_class = torch.nn.Sequential(
                torch.nn.Linear(flattened, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, self.n_outputs),
            )
