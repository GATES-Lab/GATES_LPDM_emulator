"""Encoders to encode from the input graph to the latent graph

In the original paper the encoder is described as

The Encoder maps from physical data defined on a latitude/longitude grid to abstract latent features
defined on an icosahedron grid.  The Encoder GNN uses a bipartite graph(lat/lon→icosahedron) with
edges only between nodes in the lat/lon grid and nodes in the icosahedron grid. Put another way,
spatial and channel information in the local neighborhood of each icosahedron node is
gathered using connections to nearby lat/lon nodes.

The initial node features are the 78 atmospheric variables described in Section 2.1, plus solar
radiation, orography, land-sea mask, the day-of-year,sin(lat),cos(lat),sin(lon), and cos(lon).
The initial edge features are the positions of the lat/lon nodes connected to each icosahedron node.
These positions are provided in a local coordinate system that is defined relative
to each icosahedron node.

In further notes, they notice that there is some hexagon instabilities in long rollouts
One possible way to change that is to do the additative noise as in the original MeshGraphNet
or mildly randomize graph connectivity in encoder, as a kind of edge Dropout



"""
from typing import Tuple

import einops
import h3
import numpy as np
import torch
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_adj

from model.layers.graph_net_block import MLP


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
    """Encoder graph model"""

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
        dropout=0, v2_edges=False, input_names=None, higher_res=0, idx_latlon=None, better_meshnodes=True, attention=False, release_coords="default", release_edges=False, concat_enc_neighbours=False, initial_enc=False

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
        # lat_nodes = torch.zeros((len(lat_lons_heights), input_dim), dtype=torch.float)
        # h3_nodes = torch.zeros((h3.num_hexagons(resolution), output_dim), dtype=torch.float)
        ## changed here
        #nodes = torch.zeros(
        #    (len(lat_lons) + self.num_h3, input_dim), dtype=torch.float
        #)
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
            print(np.shape(both_features))
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
        #print("encoder: setting up node enc")
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
        #print("blubububu")
        #print("features", np.shape(features))
        #print('This is being used - encoder')
        #import ipdb; ipdb.set_trace()
        batch_size = features.shape[0]
        self.batch_size = batch_size
        #print(features.device)
        self.h3_nodes = torch.nn.Parameter(self.h3_nodes.to(features.device))
        self.graph = self.graph.to(features.device) 
        self.mesh_graph = self.mesh_graph.to(features.device) 
        self.edge_weights = self.edge_weights.to(features.device)
        if self.attention:
            self.attention_mask = self.attention_mask.to(features.device)
        #print(features.size(), torch.flatten(torch.from_numpy(self.edge_weights)).size())

        if self.initial_enc:
            features = einops.rearrange(features, "b n f -> (b n) f")
            features = self.initial_encoder(features)
            features = einops.rearrange(features, "(b n) f -> b n f", b=batch_size)
        
        features = einops.rearrange(features, "b n f -> b f n")

        if self.concat_enc_neighbours:
            #print(features.size(), self.graph.edge_index.size())

            features = concat_group_by(features, self.graph.edge_index[1,:].squeeze())

            #print(features.size(),features.dtype )

        else:
            #print('features',features.shape)
            #print('edge weights',self.edge_weights.shape)
            features = torch.multiply(features, torch.flatten(self.edge_weights))

            #print("after weighting", np.shape(features))
            #print(features.size(), self.graph.edge_index[1,:].size())
            # scatter changes the shape from (b,f,latlonnodes) to (b,f,meshnodes), sorted by meshnode index 
            #print(features.size(),features.dtype )
            
            features = scatter_mean(src=features, index=self.graph.edge_index[1,:])
            #print('scatter features',features)
            #print(features.size(),features.dtype )
            #print("scatter in encoder", self.graph.edge_index[1,:])
            #print(features.size())
            #print("after scattering", np.shape(features))

        features = einops.rearrange(features, "b f n -> (b n) f")
        #print(features.size(),features.dtype )
        #print('before out',features)
        out = self.node_encoder(features)
        #print('after out',out)  
        #out = einops.rearrange(out, "(b n) f -> b n f", b=self.batch_size)
        #print("after encoding", np.shape(out))

        if self.better_meshnodes:
            self.improved_mesh_nodes = self.improved_mesh_nodes.to(features.device)
            better_nodes = einops.repeat(self.improved_mesh_nodes, "e f -> (repeat e) f", repeat=batch_size)

            #print(features.size(), out.dtype, self.improved_mesh_nodes.size(), better_nodes.dtype, better_nodes.device)
            out = torch.cat([out, better_nodes], dim=1)
        
        
        if self.v2_edges:
            # take index of wind (actually do above!)
            # take mean of wind at each two vectors
            # concat with distance/latlon difference? 
            # encode
            mesh_edge_attrs = self.mesh_edge_encoder(self.mesh_graph.edge_attr)
        else:
            mesh_edge_attrs = self.mesh_edge_encoder(self.mesh_graph.edge_attr)

        # same inputs every time... could include info from bottom nodules
        mesh_edge_attrs = einops.repeat(mesh_edge_attrs, "e f -> (repeat e) f", repeat=batch_size)
        #mesh_edge_attrs = torch.tensor(mesh_edge_attrs)
        mesh_edge_idx = torch.cat([self.mesh_graph.edge_index+ i * torch.max(self.mesh_graph.edge_index) + i for i in range(batch_size) ], dim=1)

        #mesh_edge_idx = self.mesh_graph.edge_index
#
        

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
                print("here!")
                if h3_index != self.release_h3 and (self.release_h3 not in h_points):
                    # add edge to centre, only if edge does not already exist
                    edge_targets.append(self.base_h3_map[self.release_h3])
                    edge_attrs.append([distance, loc_point[0]-loc_neighbour[0], loc_point[1]-loc_neighbour[1]])
                    edge_sources.append(self.base_h3_map[h3_index])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        return Data(edge_index=edge_index, edge_attr=edge_attrs)


class FixedSatelliteEncoder(torch.nn.Module):
    """Encoder graph model"""

    def __init__(
        self,
        lat_lons: list,
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
        dropout=0, v2_edges=False, input_names=None, higher_res=0, idx_latlon=None, better_meshnodes=True, attention=False, release_coords="default", release_edges=False

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
        if self.v2_edges:
            assert input_names is not None, "Pass input names to do edges v2 (wind on the mesh edges)"
            self.input_names=input_names

        self.release_edges = release_edges
        if self.release_edges:
            raise NotImplementedError

        
        if higher_res==0:
            # regular size grid
            self.h3_grid = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in lat_lons] 
        else:
            raise NotImplementedError
        
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
        self.h3_distances -= self.h3_distances.min()
        self.h3_distances /= self.h3_distances.max()

        # create inverse distance weights for each set of weights that goes to the same node
        self.edge_weights = np.zeros_like(self.h3_distances)
        for n in np.unique(edge_targets):
            same_edges = np.where(edge_targets==n)

            self.edge_weights[same_edges] = len(same_edges[0])*(1/self.h3_distances[same_edges])/np.sum([1/i for i in self.h3_distances[same_edges]])
        
        self.edge_weights = torch.tensor(self.edge_weights, dtype=torch.float32)


        # Build the default graph
        # lat_nodes = torch.zeros((len(lat_lons_heights), input_dim), dtype=torch.float)
        # h3_nodes = torch.zeros((h3.num_hexagons(resolution), output_dim), dtype=torch.float)
        ## changed here
        #nodes = torch.zeros(
        #    (len(lat_lons) + self.num_h3, input_dim), dtype=torch.float
        #)
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
            print(np.shape(both_features))
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
        #print("encoder: setting up node enc")
        ## this will encode the inverse distance weighted mean of all grid nodes connected to one mesh node, into a bigger dimension
        
        self.node_encoder = MLP(
            input_dim,
            output_dim,
            hidden_dim_processor_node,
            hidden_layers_processor_node,
            mlp_norm_type,
            self.use_checkpointing, dropout=dropout
        )

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
        #print(features.device)
        self.h3_nodes = torch.nn.Parameter(self.h3_nodes.to(features.device))
        self.graph = self.graph.to(features.device) 
        self.mesh_graph = self.mesh_graph.to(features.device) 
        self.edge_weights = self.edge_weights.to(features.device)
        if self.attention:
            self.attention_mask = self.attention_mask.to(features.device)
        
        features = einops.rearrange(features, "b n f -> b f n")
        #print(features.size(), torch.flatten(torch.from_numpy(self.edge_weights)).size())
        features = torch.multiply(features, torch.flatten(self.edge_weights))
        #print("after weighting", np.shape(features))
        #print(features.size(), self.graph.edge_index[1,:].size())
        # scatter changes the shape from (b,f,latlonnodes) to (b,f,meshnodes), sorted by meshnode index 
        features = scatter_mean(src=features, index=self.graph.edge_index[1,:])
        #print("scatter in encoder", self.graph.edge_index[1,:])
        #print(features.size())
        #print("after scattering", np.shape(features))
        features = einops.rearrange(features, "b f n -> (b n) f")

        out = self.node_encoder(features)  
        #out = einops.rearrange(out, "(b n) f -> b n f", b=self.batch_size)
        #print("after encoding", np.shape(out))

        if self.better_meshnodes:
            self.improved_mesh_nodes = self.improved_mesh_nodes.to(features.device)
            better_nodes = einops.repeat(self.improved_mesh_nodes, "e f -> (repeat e) f", repeat=batch_size)

            print(features.size(), out.dtype, self.improved_mesh_nodes.size(), better_nodes.dtype, better_nodes.device)
            out = torch.cat([out, better_nodes], dim=1)
        
        
        if self.v2_edges:
            # take index of wind (actually do above!)
            # take mean of wind at each two vectors
            # concat with distance/latlon difference? 
            # encode
            mesh_edge_attrs = self.mesh_edge_encoder(self.mesh_graph.edge_attr)
        else:
            mesh_edge_attrs = self.mesh_edge_encoder(self.mesh_graph.edge_attr)

        # same inputs every time... could include info from bottom nodules
        mesh_edge_attrs = einops.repeat(mesh_edge_attrs, "e f -> (repeat e) f", repeat=batch_size)
        #mesh_edge_attrs = torch.tensor(mesh_edge_attrs)
        mesh_edge_idx = torch.cat([self.mesh_graph.edge_index+ i * torch.max(self.mesh_graph.edge_index) + i for i in range(batch_size) ], dim=1)

        #mesh_edge_idx = self.mesh_graph.edge_index

        

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
                print("here!")
                if h3_index != self.release_h3 and (self.release_h3 not in h_points):
                    # add edge to centre, only if edge does not already exist
                    edge_targets.append(self.base_h3_map[self.release_h3])
                    edge_attrs.append([distance, loc_point[0]-loc_neighbour[0], loc_point[1]-loc_neighbour[1]])
                    edge_sources.append(self.base_h3_map[h3_index])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
        # Use heterogeneous graph as input and output dims are not same for the encoder
        # Because uniform grid now, don't need edge attributes as they are all the same
        return Data(edge_index=edge_index, edge_attr=edge_attrs)
