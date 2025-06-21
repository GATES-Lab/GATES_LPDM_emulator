"""Model for forecasting weather from NWP states"""
import torch
from typing import Optional
#from huggingface_hub import PyTorchModelHubMixin

from model.layers.encoder import SatelliteEncoder, FixedSatelliteEncoder
from model.layers.decoder import SatelliteDecoder, SatelliteDecoderClassifier
from model.layers.processor import SatelliteProcessor
import numpy as np


class GraphSatelliteForecaster(torch.nn.Module): #, PyTorchModelHubMixin
    """Main weather prediction model from the paper"""

    def __init__(
        self,
        lat_lons: list,
        whole_world: bool = False,
        resolution: int = 2,
        feature_dim: int = 78,
        aux_dim: int = 24, 
        output_dim: Optional[int] = None,
        node_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        residuals: bool = False,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
        dropout: float=0,
        encode_edges=True,
        encode_nodes=True,
        n_decoder_neighbours=3,
        decoder_final_layer=None,
        higher_mesh_res=0,
        idx_latlon=None,
        concat_decoder_neighbours=False, concat_decoder_neighbours_2=False, better_meshnodes=False, scatter="mean",disaggregated=False, batchsize=5, attention=False, release_coords="default", release_edges=False, decoder_append_latlon=False, concat_enc_neighbours=False,initial_enc=False
    ):
        """
        Graph Weather Model based off https://arxiv.org/pdf/2202.07575.pdf

        Args:
            lat_lons: List of latitude and longitudes for the grid
            whole_world = Use base graph for the whole world or only nodes that contain lat/lons
            NOTE: probably should change to all nodes within defined region
            resolution: Resolution of the H3 grid, prefer even resolutions, as
                odd ones have octogons and heptagons as well
            feature_dim: Input feature size
            aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc)
            output_dim: Optional, output feature size, useful if want only subset of variables in output
            node_dim: Node hidden dimension
            edge_dim: Edge hidden dimension
            num_blocks: Number of message passing blocks in the Processor
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Use gradient checkpointing to reduce model memory
        """
        
        super().__init__()
        #print(f"forecast - dropout {dropout}")
        self.feature_dim = feature_dim
        if output_dim is None:
            output_dim = self.feature_dim
        #print("set up encoder")

        self.encoder = SatelliteEncoder(
            lat_lons=lat_lons,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=feature_dim + aux_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            use_checkpointing=use_checkpointing, dropout=dropout,higher_res=higher_mesh_res,idx_latlon=idx_latlon, better_meshnodes=better_meshnodes, attention=attention, release_coords=release_coords, release_edges=release_edges, concat_enc_neighbours=concat_enc_neighbours,initial_enc=initial_enc
        )
        if not encode_edges:
            edge_dim=2
        if not encode_nodes:
            node_dim=feature_dim+aux_dim
        
        if better_meshnodes:
            node_dim=node_dim+2

        print("set up processor")
        self.processor = SatelliteProcessor(
            input_dim=node_dim,
            edge_dim=edge_dim,
            num_blocks=num_blocks,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type, dropout=dropout, scatter=scatter, disaggregated=disaggregated, attention=attention, attention_mask=self.encoder.attention_mask
        )
        print("set up decoder")
        if residuals:
            node_dim=node_dim + feature_dim + aux_dim
            print("here", node_dim)
            # if residuals, attach the original inputs for that latlon node to the inputs of the mesh nodes
        else:
            node_dim=node_dim
        self.decoder = SatelliteDecoder(
            lat_lons=lat_lons,
            h_grid=self.encoder.h3_grid,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=output_dim,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            residuals=residuals,
            hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing , dropout=dropout, final_activation=decoder_final_layer, n_neighbours=n_decoder_neighbours, concat_neighbours=concat_decoder_neighbours, concat_neighbours_2=concat_decoder_neighbours_2,idx_latlon=idx_latlon, append_latlon=decoder_append_latlon
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the new state of the forecast

        Args:
            features: The input features, aligned with the order of lat_lons_heights

        Returns:
            The next state in the forecast
        """
        #print("run encoder")
        #print("getting encoder inputs")
        #print('This is being used')
       #import ipdb; ipdb.set_trace()
        x, edge_idx, edge_attr = self.encoder(features)
        # here x has size (bxn, f) where the nodes are in order 0-max
        #print(x.size(), edge_idx.size(), edge_attr.size())
        #print("run processor", self.encoder.batch_size)
        #print("getting processor inputs")
        x = self.processor(x, edge_idx, edge_attr, batch=self.encoder.batch_size)
        #print("processed", x.size())
        #print("run decoder")
        #print(x.size())
        x = self.decoder(x, features)

        return x
    

class GraphSatelliteForecasterClassifier(torch.nn.Module): #, PyTorchModelHubMixin
    """Main weather prediction model from the paper"""
    def __init__(
        self,
        lat_lons: list,
        whole_world: bool = False,
        resolution: int = 2,
        feature_dim: int = 78,
        aux_dim: int = 24, 
        output_dim: Optional[int] = None,
        node_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        residuals: bool = False,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
        dropout: float=0,
        encode_edges=True,
        encode_nodes=True,
        n_decoder_neighbours=3,
        decoder_final_layer=None,
        higher_mesh_res=0,
        idx_latlon=None,
        concat_decoder_neighbours=False, concat_decoder_neighbours_2=False, better_meshnodes=False, scatter="mean",disaggregated=False, batchsize=5, attention=False, release_coords="default", release_edges=False, decoder_append_latlon=False, concat_enc_neighbours=False,initial_enc=False, num_classes=4
    ):
        """
        Graph Weather Model based off https://arxiv.org/pdf/2202.07575.pdf

        Args:
            lat_lons: List of latitude and longitudes for the grid
            whole_world = Use base graph for the whole world or only nodes that contain lat/lons
            NOTE: probably should change to all nodes within defined region
            resolution: Resolution of the H3 grid, prefer even resolutions, as
                odd ones have octogons and heptagons as well
            feature_dim: Input feature size
            aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc)
            output_dim: Optional, output feature size, useful if want only subset of variables in output
            node_dim: Node hidden dimension
            edge_dim: Edge hidden dimension
            num_blocks: Number of message passing blocks in the Processor
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Use gradient checkpointing to reduce model memory
        """
        
        super().__init__()
        #print(f"forecast - dropout {dropout}")
        self.feature_dim = feature_dim
        if output_dim is None:
            output_dim = self.feature_dim
        #print("set up encoder")
        # Nawid - encoder model
        self.encoder = SatelliteEncoder(
            lat_lons=lat_lons,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=feature_dim + aux_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            use_checkpointing=use_checkpointing, dropout=dropout,higher_res=higher_mesh_res,idx_latlon=idx_latlon, better_meshnodes=better_meshnodes, attention=attention, release_coords=release_coords, release_edges=release_edges, concat_enc_neighbours=concat_enc_neighbours,initial_enc=initial_enc
        )
        if not encode_edges:
            edge_dim=2
        if not encode_nodes:
            node_dim=feature_dim+aux_dim
        
        if better_meshnodes:
            node_dim=node_dim+2
            
        print("set up processor")
        # Nawid - processor model (for the hidden layers)
        self.processor = SatelliteProcessor(
            input_dim=node_dim,
            edge_dim=edge_dim,
            num_blocks=num_blocks,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type, dropout=dropout, scatter=scatter, disaggregated=disaggregated, attention=attention, attention_mask=self.encoder.attention_mask
        )

        if residuals:
            node_dim=node_dim + feature_dim + aux_dim
            #print("here", node_dim)
            # if residuals, attach the original inputs for that latlon node to the inputs of the mesh nodes
        else:
            node_dim=node_dim
        self.decoder = SatelliteDecoderClassifier(
            lat_lons=lat_lons,
            h_grid=self.encoder.h3_grid,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=output_dim,
            num_classes=num_classes,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            residuals=residuals,
            hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing , dropout=dropout, final_activation=decoder_final_layer, n_neighbours=n_decoder_neighbours, concat_neighbours=concat_decoder_neighbours, concat_neighbours_2=concat_decoder_neighbours_2,idx_latlon=idx_latlon, append_latlon=decoder_append_latlon
        )
        '''
        print("set up decoder")
        if residuals:
            node_dim=node_dim + feature_dim + aux_dim
            print("here", node_dim)
            # if residuals, attach the original inputs for that latlon node to the inputs of the mesh nodes
        else:
            node_dim=node_dim
        # Nawid - decoder model
        '''
       
        '''
        self.decoder = SatelliteDecoderClassifier(
            lat_lons=lat_lons,
            h_grid=self.encoder.h3_grid,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=output_dim,
            num_classes = num_classes,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            residuals=residuals,
            hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing , dropout=dropout, final_activation=decoder_final_layer, n_neighbours=n_decoder_neighbours, concat_neighbours=concat_decoder_neighbours
        )
        '''

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the new state of the forecast

        Args:
            features: The input features, aligned with the order of lat_lons_heights

        Returns:
            The next state in the forecast
        """
        #print("run encoder")
        #print("getting encoder inputs")
        # Nawid - encoder output
        #print('features',features)
        x, edge_idx, edge_attr = self.encoder(features)
        #print('encoder outout',x)
        # here x has size (bxn, f) where the nodes are in order 0-max
        #print(x.size(), edge_idx.size(), edge_attr.size())
        #print("run processor", self.encoder.batch_size)
        #print("getting processor inputs")
        x = self.processor(x, edge_idx, edge_attr, batch=self.encoder.batch_size)
        #print('processor outout',x)
        #print("processed", x.size())
        #print("run decoder")
        #print('before decoder',x.size())
        x = self.decoder(x, features)#
        #print('decoder output',x)

        return x


class FixedGraphSatelliteForecaster(torch.nn.Module): #, PyTorchModelHubMixin
    """Main weather prediction model from the paper"""

    def __init__(
        self,
        lat_lons: list,
        whole_world: bool = False,
        resolution: int = 2,
        feature_dim: int = 78,
        aux_dim: int = 24,
        output_dim: Optional[int] = None,
        node_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        residuals: bool = False,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
        dropout: float=0,
        encode_edges=True,
        encode_nodes=True,
        n_decoder_neighbours=3,
        decoder_final_layer=None,
        higher_mesh_res=0,
        idx_latlon=None,
        concat_decoder_neighbours=False, better_meshnodes=False, scatter="mean",disaggregated=False, batchsize=5, attention=False, release_coords="default", release_edges=False
    ):
        """
        Graph Weather Model based off https://arxiv.org/pdf/2202.07575.pdf

        Args:
            lat_lons: List of latitude and longitudes for the grid
            whole_world = Use base graph for the whole world or only nodes that contain lat/lons
            NOTE: probably should change to all nodes within defined region
            resolution: Resolution of the H3 grid, prefer even resolutions, as
                odd ones have octogons and heptagons as well
            feature_dim: Input feature size
            aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc)
            output_dim: Optional, output feature size, useful if want only subset of variables in output
            node_dim: Node hidden dimension
            edge_dim: Edge hidden dimension
            num_blocks: Number of message passing blocks in the Processor
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Use gradient checkpointing to reduce model memory
        """
        
        super().__init__()
        #print(f"forecast - dropout {dropout}")
        self.feature_dim = feature_dim
        if output_dim is None:
            output_dim = self.feature_dim
        #print("set up encoder")

        self.encoder = FixedSatelliteEncoder(
            lat_lons=lat_lons,
            resolution=resolution,
            input_dim=feature_dim + aux_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            use_checkpointing=use_checkpointing, dropout=dropout,higher_res=higher_mesh_res,idx_latlon=idx_latlon, better_meshnodes=better_meshnodes, attention=attention, release_coords=release_coords, release_edges=release_edges
        )
        if not encode_edges:
            edge_dim=2
        if not encode_nodes:
            node_dim=feature_dim+aux_dim
        
        if better_meshnodes:
            node_dim=node_dim+2

        print("set up processor")
        self.processor = SatelliteProcessor(
            input_dim=node_dim,
            edge_dim=edge_dim,
            num_blocks=num_blocks,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type, dropout=dropout, scatter=scatter, disaggregated=disaggregated, attention=attention, attention_mask=self.encoder.attention_mask
        )
        print("set up decoder")
        if residuals:
            node_dim=node_dim + feature_dim + aux_dim
            print("here", node_dim)
            # if residuals, attach the original inputs for that latlon node to the inputs of the mesh nodes
        else:
            node_dim=node_dim
        self.decoder = SatelliteDecoder(
            lat_lons=lat_lons,
            h_grid=self.encoder.h3_grid,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=output_dim,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            residuals=residuals,
            hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing , dropout=dropout, final_activation=decoder_final_layer, n_neighbours=n_decoder_neighbours, concat_neighbours=concat_decoder_neighbours
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the new state of the forecast

        Args:
            features: The input features, aligned with the order of lat_lons_heights

        Returns:
            The next state in the forecast
        """
        #print("run encoder")
        #print("getting encoder inputs")
        x, edge_idx, edge_attr = self.encoder(features)
        # here x has size (bxn, f) where the nodes are in order 0-max
        #print(x.size(), edge_idx.size(), edge_attr.size())
        #print("run processor", self.encoder.batch_size)
        #print("getting processor inputs")
        x = self.processor(x, edge_idx, edge_attr, batch=self.encoder.batch_size)
        #print("processed", x.size())
        #print("run decoder")
        #print(x.size())
        x = self.decoder(x, features)

        return x


class SequentialGraphSatelliteForecaster(torch.nn.Module): #, PyTorchModelHubMixin
    """Main weather prediction model from the paper"""

    def __init__(
        self,
        lat_lons: list,
        whole_world: bool = False,
        resolution: int = 2,
        feature_dim: int = 78,
        aux_dim: int = 24,
        output_dim: Optional[int] = None,
        node_dim: int = 256,
        edge_dim: int = 256,
        num_blocks: int = 9,
        hidden_dim_processor_node: int = 256,
        hidden_dim_processor_edge: int = 256,
        hidden_layers_processor_node: int = 2,
        hidden_layers_processor_edge: int = 2,
        hidden_dim_decoder: int = 128,
        hidden_layers_decoder: int = 2,
        residuals: bool = False,
        norm_type: str = "LayerNorm",
        use_checkpointing: bool = False,
        dropout: float=0,
        encode_edges=True,
        encode_nodes=True,
        n_decoder_neighbours=3,
        decoder_final_layer=None,
    ):
        """
        Graph Weather Model based off https://arxiv.org/pdf/2202.07575.pdf

        Args:
            lat_lons: List of latitude and longitudes for the grid
            whole_world = Use base graph for the whole world or only nodes that contain lat/lons
            NOTE: probably should change to all nodes within defined region
            resolution: Resolution of the H3 grid, prefer even resolutions, as
                odd ones have octogons and heptagons as well
            feature_dim: Input feature size
            aux_dim: Number of non-NWP features (i.e. landsea mask, lat/lon, etc)
            output_dim: Optional, output feature size, useful if want only subset of variables in output
            node_dim: Node hidden dimension
            edge_dim: Edge hidden dimension
            num_blocks: Number of message passing blocks in the Processor
            hidden_dim_processor_node: Hidden dimension of the node processors
            hidden_dim_processor_edge: Hidden dimension of the edge processors
            hidden_layers_processor_node: Number of hidden layers in the node processors
            hidden_layers_processor_edge: Number of hidden layers in the edge processors
            hidden_dim_decoder:Number of hidden dimensions in the decoder
            hidden_layers_decoder: Number of layers in the decoder
            norm_type: Type of norm for the MLPs
                one of 'LayerNorm', 'GraphNorm', 'InstanceNorm', 'BatchNorm', 'MessageNorm', or None
            use_checkpointing: Use gradient checkpointing to reduce model memory
        """
        
        super().__init__()
        #print(f"forecast - dropout {dropout}")
        self.feature_dim = feature_dim +1
        if output_dim is None:
            output_dim = self.feature_dim
        #print("set up encoder")

        self.encoder = SatelliteEncoder(
            lat_lons=lat_lons,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=feature_dim + aux_dim,
            output_dim=node_dim,
            output_edge_dim=edge_dim,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type,
            use_checkpointing=use_checkpointing, dropout=dropout,
        )
        if not encode_edges:
            edge_dim=2
        if not encode_nodes:
            node_dim=feature_dim+aux_dim
            
        print("set up processor")
        self.processor = SatelliteProcessor(
            input_dim=node_dim,
            edge_dim=edge_dim,
            num_blocks=num_blocks,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type, dropout=dropout
        )
        print("set up decoder")
        if residuals:
            node_dim=node_dim + feature_dim + aux_dim
            print("here", node_dim)
            # if residuals, attach the original inputs for that latlon node to the inputs of the mesh nodes
        else:
            node_dim=node_dim
        self.decoder = SatelliteDecoder(
            lat_lons=lat_lons,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=output_dim,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            residuals=residuals,
            hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing , dropout=dropout, final_activation=decoder_final_layer, n_neighbours=n_decoder_neighbours
        )

    def forward(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute the new state of the forecast

        Args:
            features: The input features, aligned with the order of lat_lons_heights

        Returns:
            The next state in the forecast
        """
        #print("run encoder")
        #print("getting encoder inputs")
        n_nodes = features1.size()[1]
        half = int(np.sqrt(n_nodes)/2)
        size= int(np.sqrt(n_nodes))
        starting_dist = np.zeros_like(features1[:,:,0])
        starting_dist[:,np.ravel_multi_index((half,half), (size,size))] = 1
        starting_dist = torch.tensor(starting_dist)

        
        features1 = torch.cat([features1, starting_dist[:,:,None]], dim=-1)
        x, edge_idx, edge_attr = self.encoder(features1)
        # here x has size (bxn, f) where the nodes are in order 0-max
        #print(x.size(), edge_idx.size(), edge_attr.size())
        #print("run processor", self.encoder.batch_size)
        #print("getting processor inputs")
        x = self.processor(x, edge_idx, edge_attr, batch=self.encoder.batch_size)
        #print("processed", x.size())
        #print("run decoder")
        #print(x.size())
        x = self.decoder(x, features1)

        features2 = torch.cat([features2, x], dim=-1)
        x, edge_idx, edge_attr = self.encoder(features2)
        # here x has size (bxn, f) where the nodes are in order 0-max
        #print(x.size(), edge_idx.size(), edge_attr.size())
        #print("run processor", self.encoder.batch_size)
        #print("getting processor inputs")
        x = self.processor(x, edge_idx, edge_attr, batch=self.encoder.batch_size)
        #print("processed", x.size())
        #print("run decoder")
        #print(x.size())
        x = self.decoder(x, features2)

        return x
