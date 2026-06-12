"""Encoders to encode from the input graph to the latent graph

"""
from typing import Tuple

import einops
import h3
import numpy as np
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_adj

from .graph_net_block import MLP


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




class SatelliteEncoder(torch.nn.Module):
    """Encoder graph model

    OBSOLETE: use SatelliteDynamicEncoder for all new work.
    """

    def __init__(
        self,
        lat_lons: list,
        whole_world = False,
        resolution: int = 2,
        input_dim: int = 78,
        output_dim: int = 256,
        output_edge_dim: int = 256,
        hidden_dim_processor_node=256,
        hidden_dim_processor_edge=256,
        hidden_layers_processor_node=2,
        hidden_layers_processor_edge=2,
        mlp_norm_type="LayerNorm",
        use_checkpointing: bool = False,
        dropout=0, v2_edges=False, input_names=None, higher_res=0, idx_latlon=None, better_meshnodes=True, attention=False, release_coords="default", release_edges=False, concat_enc_neighbours=False, initial_enc=False,
    ):
        """
        Encode the lat/lon data inot the isohedron graph

        Args:
            lat_lons: List of (lat,lon) points
            whole_world = Use base graph for the whole world or only nodes that contain lat/lons
            resolution: H3 resolution level
            input_dim: Input node dimension
            output_dim: Output node dimension
            output_edge_dim: Edge dimension
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            mlp_norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Whether to use gradient checkpointing to use less memory
            release_coords - latlon coordinates of the release point
            release_edges - bool, if true connect all nodes in abstract layer to corresponding release node

        modifications to og code:
            - adapted to work in the whole world or only for the area defined by the lat lon coords
        to add
            - connect each grid node to more than one mesh node
        """

        super().__init__()

        #print("setting up encoder")
        #print(f"encoder - dropout {dropout}")
        self.use_checkpointing = use_checkpointing
        self.output_dim = output_dim
        self.num_latlons = len(lat_lons)
        self.v2_edges = v2_edges
        self.concat_enc_neighbours = concat_enc_neighbours
        self.initial_enc=initial_enc
        if self.v2_edges:
            assert input_names is not None, "Pass input names to do edges v2 (wind on the mesh edges)"
            self.input_names=input_names

        self.release_edges = release_edges
        if self.release_edges:
            if release_coords=="default":
                size = int(np.sqrt(len(lat_lons)))
                release_coords = lat_lons[np.ravel_multi_index([int(size/2), int(size/2)], (size,size))]
            else:
                release_coords = release_coords
                print("be careful! not tested with anything else than default")

            self.release_h3 = h3.geo_to_h3(release_coords[0], release_coords[1], resolution)

        print("in satellite encoder!")
        if whole_world:
            self.base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
            self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]

        else:
            if higher_res==0:
                # regular size grid
                self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]

                #self.release_h3_idx =
            else:
                assert idx_latlon is not None, "Pass idx_latlon with the x-y index of each node!"
                self.h3_grid = []
                for (lat, lon), (lat_idx, lon_idx) in zip(lat_lons, idx_latlon):
                    if abs(lat_idx) > higher_res or abs(lon_idx)>higher_res:
                        self.h3_grid.append(h3.geo_to_h3(lat, lon, resolution-1))
                    else:
                        self.h3_grid.append(h3.geo_to_h3(lat, lon, resolution))

                mode="centre_child"
                if mode=="inner":
                    # inner - any small-big overlap is replaced by big
                    for n,h in enumerate(self.h3_grid):
                        if h3.h3_get_resolution(h) == resolution:
                            parent = h3.h3_to_parent(h, resolution-1)
                            if parent in self.h3_grid:
                                self.h3_grid[n] = parent

                if mode=="centre_child":
                    # centre_child - if centre small node present, keep all smalls. Otherwise delete all smalls in incomplete big
                    removals = set({})
                    for n,h in enumerate(self.h3_grid):
                        if h3.h3_get_resolution(h) == resolution - 1:
                            center_child = h3.h3_to_center_child(h, resolution)
                            if center_child in self.h3_grid:
                                self.h3_grid[n] = h3.geo_to_h3(lat_lons[n][0], lat_lons[n][1], resolution)
                            else:
                                removals.update(h3.h3_to_children(h, resolution))
                    for n,h in enumerate(self.h3_grid):
                        if h3.h3_get_resolution(h) == resolution and h in removals:
                            parent = h3.h3_to_parent(h, resolution-1)
                            self.h3_grid[n] = parent


            self.base_h3_grid = sorted(list(set(self.h3_grid)))
            # h3 grid is an index based mapping, where the first latlon coord is in the polygon defined
            # in the first idx of h3 grid (and so on)
            ## base_h3_grid is the set of these (ie all polygons to be considered, only once)

        self.base_h3_map = {h_i: i for i, h_i in enumerate(self.base_h3_grid)}

        self.num_h3 = len(self.base_h3_grid)


        # Now have the h3 grid mapping, the one-directional graph of edges connecting lat/lon to h3 nodes
        # horizontal difference
        # Also get connections between lat nodes and h3 nodes
        # this creates a latlon map to the cell index they're closest to
        # - could add multiple nodes for one grid-cell to link to?
        self.h3_distances = []
        edge_sources = []
        edge_targets = []
        for idx, h3_point in enumerate(self.h3_grid):
            lat_lon = lat_lons[idx]
            # this calculates haversine distance, which is distance on the surface of the earth
            distance = h3.point_dist(lat_lon, h3.h3_to_geo(h3_point), unit="km")
            #self.h3_distances.append([np.sin(distance), np.cos(distance)])
            self.h3_distances.append([distance])

            edge_sources.append(idx)
            edge_targets.append(self.base_h3_map[h3_point])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        self.h3_distances = torch.tensor(self.h3_distances, dtype=torch.float)
            ## this tensor is later encoded then repeated for each batch
        # Compress to between 0 and 1
        #self.h3_distances -= self.h3_distances.min()
        #self.h3_distances /= self.h3_distances.max()

        # create inverse distance weights for each set of weights that goes to the same node
        self.edge_weights = np.zeros_like(self.h3_distances)
        for n in np.unique(edge_targets):
            same_edges = np.where(edge_targets==n)

            self.edge_weights[same_edges] = len(same_edges[0])*(1/self.h3_distances[same_edges])/np.sum([1/i for i in self.h3_distances[same_edges]])

        self.edge_weights = torch.tensor(self.edge_weights, dtype=torch.float32)


        # Build the default graph
        nodes = torch.ones(
            (len(lat_lons) + self.num_h3, input_dim), dtype=torch.float
        )

        # Use homogenous graph to make it easier
        self.graph = Data(x=nodes, edge_index=edge_index, edge_attr=self.h3_distances)


        # Extra starting ones for appending to inputs, could 'learn' good starting points
        self.h3_nodes = torch.nn.Parameter(torch.zeros((self.num_h3, input_dim), dtype=torch.float))
        self.better_meshnodes = better_meshnodes

        if self.better_meshnodes:
            """
            better meshnodes adds location/distance features to the nodes in the abstract layer (which otherwise would be empty)
            # TODO test and eval (this is only experimental)
            """
            assert idx_latlon is not None, "pass the idx_latlon grid to do better meshnodes!"
            size = int(np.sqrt(self.num_latlons))
            centre_idx = np.ravel_multi_index((int(size/2),int(size/2)), (size,size))
            print(centre_idx)
            assert idx_latlon[centre_idx] == (0,0), "something went wrong trying to make better meshnodes"

            binary_centre = np.zeros((self.num_h3,1))
            binary_centre[self.base_h3_grid.index(self.h3_grid[centre_idx])] = 1

            distance_centre = []

            for idx, h3_point in enumerate(self.base_h3_grid):
                # this calculates haversine distance, which is distance on the surface of the earth
                distance = h3.point_dist(lat_lons[centre_idx], h3.h3_to_geo(h3_point), unit="km")
                distance_centre.append([distance])

            distance_centre = np.array(distance_centre)
            distance_centre = (distance_centre-np.min(distance_centre))/(np.max(distance_centre)-np.min(distance_centre))

            both_features = np.hstack((binary_centre, distance_centre))
            #print(np.shape(both_features))
            self.improved_mesh_nodes = torch.tensor(both_features, dtype=torch.float32)

        self.mesh_graph = self.create_mesh_graph()

        self.attention=attention
        if attention:
            """
            Attention mask means attention is only applied to connected nodes
            """
            self.attention_mask = 1-to_dense_adj(self.mesh_graph.edge_index).squeeze()
            self.attention_mask.fill_diagonal_(1)
        else:
            self.attention_mask = None

        # Output graph
        ## this will encode the inverse distance weighted mean of all grid nodes connected to one mesh node, into a bigger dimension
        node_encoder_input_dim = input_dim

        if self.initial_enc:
            self.initial_encoder = MLP(
                    input_dim,
                    output_dim,
                    hidden_dim_processor_node,
                    hidden_layers_processor_node,
                    mlp_norm_type,
                    self.use_checkpointing, dropout=dropout
                )
            node_encoder_input_dim = output_dim

        if self.concat_enc_neighbours:
            node_encoder_input_dim = np.max(np.unique(self.graph.edge_index[1], return_counts=True)[1])* node_encoder_input_dim

        self.node_encoder = MLP(
            node_encoder_input_dim,
            output_dim,
            hidden_dim_processor_node,
            hidden_layers_processor_node,
            mlp_norm_type,
            self.use_checkpointing, dropout=dropout
        )
        #print("mesh edge encoder inputs:", self.mesh_graph.edge_attr.size()[-1])
        self.mesh_edge_encoder = MLP(
            self.mesh_graph.edge_attr.size()[-1],
            output_edge_dim,
            hidden_dim_processor_edge,
            hidden_layers_processor_edge,
            mlp_norm_type,
            self.use_checkpointing, dropout=dropout
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adds features to the encoding graph

        Args:
            features: Array of features in same order as lat_lon

        Returns:
            Torch tensors of node features, latent graph edge index, and latent edge attributes
        """
        batch_size = features.shape[0]
        self.batch_size = batch_size
        self.h3_nodes = torch.nn.Parameter(self.h3_nodes.to(features.device))
        self.graph = self.graph.to(features.device)
        self.mesh_graph = self.mesh_graph.to(features.device)
        self.edge_weights = self.edge_weights.to(features.device)
        if self.attention:
            self.attention_mask = self.attention_mask.to(features.device)

        if self.initial_enc:
            features = einops.rearrange(features, "b n f -> (b n) f")
            features = self.initial_encoder(features)
            features = einops.rearrange(features, "(b n) f -> b n f", b=batch_size)

        features = einops.rearrange(features, "b n f -> b f n")
        #print("features shape before enc and rearrange:", features.shape)

        if self.concat_enc_neighbours:
            features = concat_group_by(features, self.graph.edge_index[1,:].squeeze())
        else:
            features = torch.multiply(features, torch.flatten(self.edge_weights))
            features = scatter_mean(src=features, index=self.graph.edge_index[1,:])

        features = einops.rearrange(features, "b f n -> (b n) f")
        out = self.node_encoder(features)

        #print("out shape after node encoder:", out.shape)

        if self.better_meshnodes:
            self.improved_mesh_nodes = self.improved_mesh_nodes.to(features.device)
            better_nodes = einops.repeat(self.improved_mesh_nodes, "e f -> (repeat e) f", repeat=batch_size)
            out = torch.cat([out, better_nodes], dim=1)

        if self.v2_edges:
            mesh_edge_attrs = self.mesh_edge_encoder(self.mesh_graph.edge_attr)
        else:
            mesh_edge_attrs = self.mesh_edge_encoder(self.mesh_graph.edge_attr)
        
        # print mesh_graph edge index and edge attr shapes
        # print("mesh_graph edge_index shape:", self.mesh_graph.edge_index.shape)
        # print("mesh_graph edge_attr shape:", self.mesh_graph.edge_attr.shape)

        # same inputs every time... could include info from bottom nodules
        mesh_edge_attrs = einops.repeat(mesh_edge_attrs, "e f -> (repeat e) f", repeat=batch_size)
        mesh_edge_idx = torch.cat([self.mesh_graph.edge_index+ i * torch.max(self.mesh_graph.edge_index) + i for i in range(batch_size) ], dim=1)
        #print("mesh_edge_idx shape:", mesh_edge_idx.shape)
        #print("mesh_edge_attrs shape:", mesh_edge_attrs.shape)
        #print("out shape:", out.shape)
        # print out[:5, :5], mesh_edge_idx[:, :5], mesh_edge_attrs[:5, :5]
        #print("out[:5, :5]", out[:5, :5])
        #print("mesh_edge_idx[:, :5]", mesh_edge_idx[:, :5])
        #print("mesh_edge_attrs[:5, :5]", mesh_edge_attrs[:5, :5])
        return (
            out,
            mesh_edge_idx,
            mesh_edge_attrs
        )  # New graph

    def create_mesh_graph(self) -> Data:
        """
        Copies over and generates a Data object for the processor to use

        Returns:
            The connectivity and edge attributes for the latent graph
        """
        # Get connectivity of the graph
        edge_sources = []
        edge_targets = []
        edge_attrs = []
        for h3_index in self.base_h3_grid:
            # itself and all one-hop neighbouring points
            h_points = h3.k_ring(h3_index, 1)
            loc_point = h3.h3_to_geo(h3_index)
            for h in h_points:
                loc_neighbour = h3.h3_to_geo(h)
                distance = h3.point_dist(loc_point, loc_neighbour, unit="km")
                try:
                    edge_targets.append(self.base_h3_map[h])
                    edge_attrs.append([distance, loc_point[0]-loc_neighbour[0], loc_point[1]-loc_neighbour[1]])
                    edge_sources.append(self.base_h3_map[h3_index])
                except KeyError:
                    # this except will be triggered if h (one of the neighbouring points to h3_index) is not in the list
                    # this can only happen if using a reduced domain (whole_world = False), at the edges of this domain
                    continue
            if self.release_edges:
                print("here in release edges")
                if h3_index != self.release_h3 and (self.release_h3 not in h_points):
                    # add edge to centre, only if edge does not already exist
                    edge_targets.append(self.base_h3_map[self.release_h3])
                    edge_attrs.append([distance, loc_point[0]-loc_neighbour[0], loc_point[1]-loc_neighbour[1]])
                    edge_sources.append(self.base_h3_map[h3_index])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        #print(edge_index[:,:5])
        #print(edge_attrs[:5, :5])
        return Data(edge_index=edge_index, edge_attr=edge_attrs)




class SatelliteDynamicEncoder(torch.nn.Module):
    """Encoder with optional dynamic wind and/or absolute lat/lon features on mesh edges.

    Update to SatelliteEncoder, with mesh edge encondings that are extracted from the data.  wind_mesh_edges=True adds the wind variables determined by wind_indices as the mean of the two endpoint mesh-node feature vectors. latlon_mesh_edges=True adds the delta_lat and delta_lon between the two endpoint mesh-node feature vectors. Note that as this is calculated from the data, if the lat/lon has been normalised, the dlat and dlon will be calculated in the normalised space.
    dynamic_earthdistance=True replaces the static haversine distance with a dynamic one calculated from the lat/lon features. When dynamic_earthdistance=True, the lat/lon features are removed (by replacing them with random noise)
    All core behaviour is identical to SatelliteEncoder when both flags (latlon_mesh_edges and wind_indices) are False.

    Improvements over SatelliteEncoder:
    - Graph tensors are registered as buffers — no .to(device) calls in forward().
    - Batched edge index uses vectorised broadcasting instead of a Python loop.
    - Dead self.h3_nodes parameter is not included.

    Args:
        lat_lons: List of (lat,lon) points
        whole_world = Use base graph for the whole world or only nodes that contain lat/lons
        resolution: H3 resolution level
        input_dim: Input node dimension
        output_dim: Output node dimension
        output_edge_dim: Edge dimension
        hidden_dim_processor_node: Hidden dimension of the node processors
        hidden_dim_processor_edge: Hidden dimension of the edge processors
        hidden_layers_processor_node: Number of hidden layers in the node processors
        hidden_layers_processor_edge: Number of hidden layers in the edge processors
        mlp_norm_type: Type of norm for the MLPs
            one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
        use_checkpointing: Whether to use gradient checkpointing to use less memory
        dropout: Dropout probability. 0 means no dropout.
        higher_res: (do not use, needs testing)
        idx_latlon: List of (lat_idx, lon_idx) integer tuples for each node, indicating the x and y index of the node in the original lat-lon grid. Required if higher_res > 0 or better_meshnodes=True.
        better_meshnodes: (do not use, needs testing)
        attention: Whether to use attention in the processor. If True, an attention mask is created so that nodes only attend to their connected neighbours. (needs testing)
        release_edges: bool, if true connect all nodes in abstract layer to corresponding release node. (needs testing)
        release_coords: latlon coordinates of the release point, required if release_edges=True. If "default", uses the centre point of the latlon grid. (needs testing)
        concat_enc_neighbours: Whether to concatenate (instead of mean) the features of the grid nodes connected to each mesh node before encoding. Cannot be combined with dynamic_edges. (needs testing)
        initial_enc: Whether to apply an initial encoding MLP to the input features before any aggregation. Cannot be combined with dynamic_edges. (needs testing)
        wind_mesh_edges (bool): Whether to add wind features to the mesh edges. 
        wind_indices (list of ints): Indices of the input feature tensor corresponding to the wind variables to use for the mesh edge features. The mesh attribute is calculated as the mean of the wind features at the two endpoint mesh nodes. Required if wind_mesh_edges=True. Cannot be combined with concat_enc_neighbours or initial_enc.
        latlon_mesh_edges (bool): Whether to add lat/lon features to the mesh edges. The mesh edge attributes are extended with the delta_lat and delta_lon between the two endpoint mesh nodes, calculated from the input features at the latlon_indices. 
        latlon_indices (list of 2 ints): Indices of the input feature tensor corresponding to the latitude and longitude variables to use for calculating the lat/lon features for the mesh edges. Required if latlon_mesh_edges=True. 
        dynamic_earthdistance (bool): Whether to replace the static haversine distance edge attribute with a dynamic one calculated from the lat/lon features. When True, the lat/lon features are removed from the input (replaced with random noise) and the mesh edge attributes are extended with a dynamic earth distance calculated from the lat/lon features at the latlon_indices. Requires latlon_mesh_edges=True.

    """

    def __init__(
        self,
        lat_lons: list,
        whole_world: bool = False,
        resolution: int = 2,
        input_dim: int = 78,
        output_dim: int = 256,
        output_edge_dim: int = 256,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        mlp_norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
        dropout: float = 0,
        higher_res: int = 0,
        idx_latlon=None,
        better_meshnodes: bool = False,
        attention: bool = False,
        release_coords="default",
        release_edges: bool = False,
        concat_enc_neighbours: bool = False,
        initial_enc: bool = False,
        wind_mesh_edges: bool = False,
        wind_indices=None,
        latlon_mesh_edges: bool = False,
        latlon_indices=None,
        dynamic_earthdistance:bool=False
    ):
    
        super().__init__()

        print("in satellite dynamic encoder!")

        dynamic_edges = wind_mesh_edges or latlon_mesh_edges

        if dynamic_edges and initial_enc:
            raise ValueError(
                "dynamic_edges and initial_enc cannot both be True: "
                "wind_indices refer to raw input features, not the encoded ones."
            )
        if dynamic_edges and concat_enc_neighbours:
            raise ValueError(
                "dynamic_edges and concat_enc_neighbours cannot both be True: "
                "the scatter is replaced by concat_group_by and wind_indices would be invalid."
            )
        if wind_mesh_edges and wind_indices is None:
            raise ValueError("wind_indices must be provided when wind_mesh_edges=True.")
        if latlon_mesh_edges and latlon_indices is None:
            raise ValueError("latlon_indices must be provided when latlon_mesh_edges=True.")
        if latlon_mesh_edges and len(latlon_indices) != 2:
            raise ValueError(f"latlon_indices must have exactly 2 entries [lat_idx, lon_idx], got {len(latlon_indices)}.")

        self.use_checkpointing = use_checkpointing
        self.output_dim = output_dim
        self.num_latlons = len(lat_lons)
        self.concat_enc_neighbours = concat_enc_neighbours
        self.initial_enc = initial_enc
        self.wind_mesh_edges = wind_mesh_edges
        self.latlon_mesh_edges = latlon_mesh_edges
        if dynamic_earthdistance and not latlon_mesh_edges:
            print("Warning: dynamic_earthdistance=True has no effect when latlon_mesh_edges=False.")
            self.dynamic_earthdistance = False
        else:
            self.dynamic_earthdistance = dynamic_earthdistance

        if wind_mesh_edges:
            self.register_buffer('wind_indices', torch.tensor(wind_indices, dtype=torch.long))
        if latlon_mesh_edges:
            self.register_buffer('latlon_indices', torch.tensor(latlon_indices, dtype=torch.long))

        # --- release point ---
        self.release_edges = release_edges
        if self.release_edges:
            if release_coords == "default":
                size = int(np.sqrt(len(lat_lons)))
                release_coords = lat_lons[np.ravel_multi_index(
                    [int(size / 2), int(size / 2)], (size, size))]
            self.release_h3 = h3.geo_to_h3(release_coords[0], release_coords[1], resolution)

        # --- build h3 grid ---
        if whole_world:
            self.base_h3_grid = sorted(list(h3.uncompact(h3.get_res0_indexes(), resolution)))
            self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
        else:
            if higher_res == 0:
                self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons]
            else:
                assert idx_latlon is not None, "Pass idx_latlon with the x-y index of each node!"
                self.h3_grid = []
                for (lat, lon), (lat_idx, lon_idx) in zip(lat_lons, idx_latlon):
                    if abs(lat_idx) > higher_res or abs(lon_idx) > higher_res:
                        self.h3_grid.append(h3.geo_to_h3(lat, lon, resolution - 1))
                    else:
                        self.h3_grid.append(h3.geo_to_h3(lat, lon, resolution))

                mode = "centre_child"
                if mode == "inner":
                    for n, h_cell in enumerate(self.h3_grid):
                        if h3.h3_get_resolution(h_cell) == resolution:
                            parent = h3.h3_to_parent(h_cell, resolution - 1)
                            if parent in self.h3_grid:
                                self.h3_grid[n] = parent
                if mode == "centre_child":
                    removals = set()
                    for n, h_cell in enumerate(self.h3_grid):
                        if h3.h3_get_resolution(h_cell) == resolution - 1:
                            center_child = h3.h3_to_center_child(h_cell, resolution)
                            if center_child in self.h3_grid:
                                self.h3_grid[n] = h3.geo_to_h3(
                                    lat_lons[n][0], lat_lons[n][1], resolution)
                            else:
                                removals.update(h3.h3_to_children(h_cell, resolution))
                    for n, h_cell in enumerate(self.h3_grid):
                        if h3.h3_get_resolution(h_cell) == resolution and h_cell in removals:
                            self.h3_grid[n] = h3.h3_to_parent(h_cell, resolution - 1)

            self.base_h3_grid = sorted(list(set(self.h3_grid)))

        self.base_h3_map = {h_i: i for i, h_i in enumerate(self.base_h3_grid)}
        self.num_h3 = len(self.base_h3_grid)

        # --- grid-to-mesh edges ---
        h3_distances = []
        edge_sources = []
        edge_targets = []
        for idx, h3_point in enumerate(self.h3_grid):
            lat_lon = lat_lons[idx]
            distance = h3.point_dist(lat_lon, h3.h3_to_geo(h3_point), unit="km")
            h3_distances.append([distance])
            edge_sources.append(idx)
            edge_targets.append(self.base_h3_map[h3_point])

        self.register_buffer('enc_edge_index',
                             torch.tensor([edge_sources, edge_targets], dtype=torch.long))

        h3_distances_arr = np.array(h3_distances)
        edge_weights = np.zeros_like(h3_distances_arr)
        for n in np.unique(edge_targets):
            same = np.where(np.array(edge_targets) == n)
            edge_weights[same] = (
                len(same[0])
                * (1 / h3_distances_arr[same])
                / np.sum(1 / h3_distances_arr[same])
            )
        self.register_buffer('enc_edge_weights',
                             torch.tensor(edge_weights, dtype=torch.float32))

        # --- better_meshnodes ---
        self.better_meshnodes = better_meshnodes
        if self.better_meshnodes:
            assert idx_latlon is not None, "pass the idx_latlon grid to do better meshnodes!"
            size = int(np.sqrt(self.num_latlons))
            centre_idx = np.ravel_multi_index((int(size / 2), int(size / 2)), (size, size))
            assert idx_latlon[centre_idx] == (0, 0), \
                "something went wrong trying to make better meshnodes"

            binary_centre = np.zeros((self.num_h3, 1))
            binary_centre[self.base_h3_grid.index(self.h3_grid[centre_idx])] = 1

            distance_centre = []
            for _, h3_point in enumerate(self.base_h3_grid):
                distance = h3.point_dist(lat_lons[centre_idx], h3.h3_to_geo(h3_point), unit="km")
                distance_centre.append([distance])
            distance_centre = np.array(distance_centre)
            distance_centre = ((distance_centre - np.min(distance_centre))
                               / (np.max(distance_centre) - np.min(distance_centre)))

            self.register_buffer('improved_mesh_nodes',
                                 torch.tensor(np.hstack((binary_centre, distance_centre)),
                                              dtype=torch.float32))

        # --- mesh graph ---
        mesh_edge_index, mesh_edge_attr_static = self._create_mesh_edges()
        self.register_buffer('mesh_edge_index', mesh_edge_index)
        self.register_buffer('mesh_edge_attr_static', mesh_edge_attr_static)

        # --- attention ---
        self.attention = attention
        if attention:
            attn_mask = 1 - to_dense_adj(mesh_edge_index).squeeze()
            attn_mask.fill_diagonal_(1)
            self.register_buffer('attention_mask', attn_mask)
        else:
            self.attention_mask = None

        # --- node encoder MLP ---
        node_encoder_input_dim = input_dim
        if self.initial_enc:
            self.initial_encoder = MLP(
                input_dim, output_dim, hidden_dim_processor_node,
                hidden_layers_processor_node, mlp_norm_type,
                self.use_checkpointing, dropout=dropout,
            )
            node_encoder_input_dim = output_dim
        if self.concat_enc_neighbours:
            node_encoder_input_dim = (
                int(np.max(np.unique(
                    self.enc_edge_index[1].numpy(), return_counts=True)[1]))
                * node_encoder_input_dim
            )
        self.node_encoder = MLP(
            node_encoder_input_dim, output_dim, hidden_dim_processor_node,
            hidden_layers_processor_node, mlp_norm_type,
            self.use_checkpointing, dropout=dropout,
        )

        # --- mesh edge encoder MLP ---
        base_edge_dim = 3 # distance, dlat and dlon from fixed grid
        if latlon_mesh_edges:
            if self.dynamic_earthdistance:
                base_edge_dim = 3  # replace with dynamic haversine distance, dlat and dlon from data
            else:
                base_edge_dim = 2  # replace with dlat and dlon from data

        if self.dynamic_earthdistance:
            from gates.data import haversine
            # calculate mean distance between mesh centres from library h3
            # using formula to derive the apothem (distance from the center to the flat midpoint of any side) from the side length, and multiplying by 2 to get the distance between centers of adjacent hexagons
            mean_edge_length = h3.edge_length(resolution, unit='km')*(np.sqrt(3)/2)*2
            self.register_buffer('mean_edge_length', torch.tensor(mean_edge_length, dtype=torch.float32))

        n_wind = len(wind_indices) if wind_mesh_edges else 0 # append as many features as there are wind variables selected
        print("mesh edge encoder inputs:", base_edge_dim + n_wind)
        print(f"base edge dim: {base_edge_dim }, with {latlon_mesh_edges=}, {self.dynamic_earthdistance=} and {wind_mesh_edges=} with {n_wind} wind features")

        self.mesh_edge_encoder = MLP(
            base_edge_dim + n_wind, output_edge_dim, hidden_dim_processor_edge,
            hidden_layers_processor_edge, mlp_norm_type,
            self.use_checkpointing, dropout=dropout,
        )

    def _create_mesh_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build mesh edge index and static edge attribute tensor.

        Returns:
            edge_index: (2, E) long tensor
            edge_attr:  (E, 3) float tensor [distance, dlat, dlon]
            
        """
        edge_sources, edge_targets, edge_attrs = [], [], []
        for h3_index in self.base_h3_grid:
            h_points = h3.k_ring(h3_index, 1)
            loc_point = h3.h3_to_geo(h3_index)
            for h_cell in h_points:
                loc_neighbour = h3.h3_to_geo(h_cell)
                distance = h3.point_dist(loc_point, loc_neighbour, unit="km")
                try:
                    # if h_cell is outside the reduced domain (edge of mesh), it won't be in base_h3_map and we skip it
                    edge_targets.append(self.base_h3_map[h_cell])
                    edge_sources.append(self.base_h3_map[h3_index])

                    dlat = loc_point[0] - loc_neighbour[0]
                    dlon = loc_point[1] - loc_neighbour[1]
                    edge_attrs.append([distance, dlat, dlon])
                except KeyError:
                    # h_cell is outside the reduced domain (edge of mesh)
                    #print(f"h_cell {h_cell} not in base_h3_map, skipping edge from {h3_index} to {h_cell}")
                    continue
            if self.release_edges:
                if h3_index != self.release_h3 and (self.release_h3 not in h_points):
                    release_loc = h3.h3_to_geo(self.release_h3)
                    dist_to_release = h3.point_dist(loc_point, release_loc, unit="km")
                    dlat = loc_point[0] - release_loc[0]
                    dlon = loc_point[1] - release_loc[1]
                    edge_attrs.append([dist_to_release, dlat, dlon])
                    edge_targets.append(self.base_h3_map[self.release_h3])
                    edge_sources.append(self.base_h3_map[h3_index])

        #print("mesh edge index shape:", (len(edge_sources),))
        #print("mesh edge attr shape:", (len(edge_attrs), len(edge_attrs[0]) if edge_attrs else 0))
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
        #print(edge_index[:,:5])
        #print(edge_attrs[:5, :5])

        return (edge_index,
                edge_attrs)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, n_nodes, feature_dim)

        Returns:
            out:             (batch * n_mesh_nodes, node_dim)
            mesh_edge_idx:   (2, batch * E)
            mesh_edge_attrs: (batch * E, edge_dim)
        """
        batch_size = features.shape[0]
        self.batch_size = batch_size  # read by processor via self.encoder.batch_size

        if self.initial_enc:
            features = einops.rearrange(features, "b n f -> (b n) f")
            features = self.initial_encoder(features)
            features = einops.rearrange(features, "(b n) f -> b n f", b=batch_size)

        features = einops.rearrange(features, "b n f -> b f n")
        ### print shape of features
        #print("features shape after initial enc and rearrange:", features.shape)

        if self.concat_enc_neighbours:
            features = concat_group_by(features, self.enc_edge_index[1, :].squeeze())
        else:
            features = torch.multiply(features, torch.flatten(self.enc_edge_weights))
            features = scatter_mean(src=features, index=self.enc_edge_index[1, :])
        # features: (batch, n_features, n_mesh_nodes)
        #print("features shape after scatter:", features.shape)
        # extract dynamic edge features before rearranging to flat (b*n, f)
        # features is still (B, n_features, n_mesh_nodes) here
        if self.wind_mesh_edges:
            src_wind = features[:, self.wind_indices][:, :, self.mesh_edge_index[0, :]]
            dst_wind = features[:, self.wind_indices][:, :, self.mesh_edge_index[1, :]]
            # mean wind across the two endpoint mesh nodes: (B, n_wind, E)
            edge_wind = (src_wind + dst_wind) / 2
            edge_wind = einops.rearrange(edge_wind, "b f e -> (b e) f")  # (B*E, n_wind)

        if self.latlon_mesh_edges:
            src_latlon = features[:, self.latlon_indices][:, :, self.mesh_edge_index[0, :]]  # (B, 2, E)
            dst_latlon = features[:, self.latlon_indices][:, :, self.mesh_edge_index[1, :]]  # (B, 2, E)
            edge_latlon = src_latlon - dst_latlon                                            # (B, 2, E): [dlat, dlon]

            if self.dynamic_earthdistance:    
                # calculate haversine distance for each edge using the lat/lon of the endpoints
                from gates.data import haversine
                src_lat = features[:, self.latlon_indices[1], :][:, None, self.mesh_edge_index[0, :]]  # (B, 1, E)
                src_lon = features[:, self.latlon_indices[1], :][:, None, self.mesh_edge_index[0, :]]  # (B, 1, E)
                dst_lat = features[:, self.latlon_indices[0], :][:, None, self.mesh_edge_index[1, :]]  # (B, 1, E)
                dst_lon = features[:, self.latlon_indices[1], :][:, None, self.mesh_edge_index[1, :]]  # (B, 1, E)

                edge_distance = haversine(src_lat.squeeze(1), src_lon.squeeze(1),
                                           dst_lat.squeeze(1), dst_lon.squeeze(1), degrees=True)  # (B, E)
                # normalize by mean edge length to keep in similar range as dlat/dlon
                #print(src_lat[:,:,:3], src_lon[:,:,:3], dst_lat[:,:,:3], dst_lon[:,:,:3])
                #print(src_lat.shape, src_lat.squeeze(1).shape)
                #print(edge_distance.shape)
                #print(edge_distance[:, :20])
                edge_distance = edge_distance / self.mean_edge_length
                #print(edge_distance[:, :20])
                edge_latlon = torch.cat([edge_distance.unsqueeze(1), edge_latlon], dim=1)  # (B, 3, E)

                ## by default, replacing the lat and lon features with random, small noise to be ignored
                #print("features before replacing lat/lon with noise (showing first 2 lat/lon features):")
                #print(features[:, self.latlon_indices, :2])
                features[:, self.latlon_indices, :] = torch.randn_like(features[:, self.latlon_indices, :]) * 1e-2
                #print(features[:, self.latlon_indices, :2])

            edge_latlon = einops.rearrange(edge_latlon, "b f e -> (b e) f")                 # (B*E, 2) or (B*E, 3) if dynamic_earthdistance

        features = einops.rearrange(features, "b f n -> (b n) f")
        out = self.node_encoder(features)

        if self.better_meshnodes:
            better_nodes = einops.repeat(
                self.improved_mesh_nodes, "e f -> (repeat e) f", repeat=batch_size)
            out = torch.cat([out, better_nodes], dim=1)

        # encode mesh edges
        if self.latlon_mesh_edges or self.wind_mesh_edges:
            if self.latlon_mesh_edges:
                mesh_edge_feature_components = [edge_latlon]  # (B*E, 2) per-sample dlat/dlon, ignores static
            else:
                mesh_edge_feature_components = [einops.repeat(
                    self.mesh_edge_attr_static, "e f -> (rep e) f", rep=batch_size)]  # (B*E, 3)
            if self.wind_mesh_edges:
                mesh_edge_feature_components.append(edge_wind)  # (B*E, n_wind)
            
            edges = torch.cat(mesh_edge_feature_components, dim=-1)
            #print(f"edges: {edges.shape}")
            #print(edges)
            mesh_edge_attrs = self.mesh_edge_encoder(torch.cat(mesh_edge_feature_components, dim=-1))
        else:
            #print(f"edges: {self.mesh_edge_attr_static.shape}")
            #print(self.mesh_edge_attr_static)
            mesh_edge_attrs = self.mesh_edge_encoder(self.mesh_edge_attr_static)
        
            mesh_edge_attrs = einops.repeat(
                mesh_edge_attrs, "e f -> (rep e) f", rep=batch_size)

        #print("after encoding:"
        #      )
        #print(mesh_edge_attrs)
        # batched edge index — vectorised over batch dimension
        offset = torch.arange(batch_size, device=self.mesh_edge_index.device) * self.num_h3
        mesh_edge_idx = (self.mesh_edge_index.unsqueeze(0)
                         + offset.view(-1, 1, 1))   # (B, 2, E)
        mesh_edge_idx = mesh_edge_idx.permute(1, 0, 2).reshape(2, -1)  # (2, B*E)

        # print("mesh_edge_idx shape:", mesh_edge_idx.shape)
        # print("out[:5, :5]", out[:5, :5])
        # print("mesh_edge_idx[:, :5]", mesh_edge_idx[:, :5])
        # print("mesh_edge_attrs[:5, :5]", mesh_edge_attrs[:5, :5])

        return out, mesh_edge_idx, mesh_edge_attrs
