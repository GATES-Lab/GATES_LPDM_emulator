"""
author: Elena Fillola @elenafillo

Implementation of the GATES model, a graph-based satellite forecaster

Developed from code by Climate Fix (Copyright (c) 2022 Open Climate Fix) implementing Keisler et al.'s weather forecasting model (see https://github.com/openclimatefix/graph_weather)

"""



import torch
from typing import Optional


from .layers.encoder import SatelliteEncoder, SatelliteDynamicEncoder
from .layers.decoder import SatelliteDecoder
from .layers.processor import SatelliteProcessor
import numpy as np


class GraphSatelliteForecaster(torch.nn.Module): #, PyTorchModelHubMixin
    """GATES - Graph based footprint emulator."""

    def __init__(
        self,
        lat_lons: list,
        whole_world: bool = False,
        resolution: int = 2,
        feature_dim: int = 78,
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
        concat_decoder_neighbours=False, concat_decoder_neighbours_2=False, scatter="mean",disaggregated=False, batchsize=5, attention=False, release_coords="default", release_edges=False, decoder_append_latlon=False, concat_enc_neighbours=False,initial_enc=False, initial_enc_dim=None,
        wind_mesh_edges=False,
        wind_indices=None,
        latlon_mesh_edges=False,
        latlon_indices=None,
        dynamic_earthdistance=False,
        use_dynamic_encoder=True
    ):
        """Initialize GATES.

        GATES uses a GNN in an encode-process-decode architecture to output footprint
        values. The original model architecture was based on Keisler's weather
        forecasting setup (https://arxiv.org/pdf/2202.07575.pdf).

        Args:
            lat_lons (list): List of latitude and longitudes for the grid.
            whole_world (bool, optional): Use base graph for the whole world or only
                nodes that contain lat/lons. NOTE: probably should change to all
                nodes within defined region. Defaults to False.
            resolution (int, optional): Resolution of the H3 grid, prefer even
                resolutions, as odd ones have octagons and heptagons as well.
                Defaults to 2.
            feature_dim (int, optional): Input feature size. Defaults to 78.
            output_dim (int, optional): Output feature size, useful if you want only
                a subset of variables in the output. If None, defaults to
                ``feature_dim``. Defaults to None.
            node_dim (int, optional): Node hidden dimension. Defaults to 256.
            edge_dim (int, optional): Edge hidden dimension. Defaults to 256.
            num_blocks (int, optional): Number of message passing blocks in the
                Processor. Defaults to 9.
            hidden_dim_processor_node (int, optional): Hidden dimension of the node
                processors. Defaults to 256.
            hidden_dim_processor_edge (int, optional): Hidden dimension of the edge
                processors. Defaults to 256.
            hidden_layers_processor_node (int, optional): Number of hidden layers in
                the node processors. Defaults to 2.
            hidden_layers_processor_edge (int, optional): Number of hidden layers in
                the edge processors. Defaults to 2.
            hidden_dim_decoder (int, optional): Number of hidden dimensions in the
                decoder. Defaults to 128.
            hidden_layers_decoder (int, optional): Number of layers in the decoder.
                Defaults to 2.
            residuals (bool, optional): If True, the processor's node update is a
                residual one (each mesh-node message-passing block adds its output to
                the previous node state rather than replacing it); forwarded to
                ``SatelliteProcessor``. Defaults to False.
            norm_type (str, optional): Type of norm for the MLPs, one of
                "LayerNorm", "GraphNorm", "InstanceNorm", "BatchNorm", "MessageNorm",
                or None. Defaults to "LayerNorm".
            use_checkpointing (bool, optional): Use gradient checkpointing to reduce
                model memory. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.
            encode_edges (bool, optional): If False, skips the edge encoder and uses
                a fixed edge dimension of 2 for the processor instead. Defaults to True.
            encode_nodes (bool, optional): If False, skips the node encoder and uses
                ``feature_dim`` as the processor's node dimension instead. Defaults to True.
            n_decoder_neighbours (int, optional): Number of closest mesh nodes each
                latlon point is connected to in the decoder; forwarded to
                ``SatelliteDecoder`` as ``n_neighbours``. Defaults to 3.
            decoder_final_layer (optional): Final activation for the decoder;
                forwarded to ``SatelliteDecoder`` as ``final_activation``. Defaults to None.
            higher_mesh_res (int, optional): Forwarded to the encoder as ``higher_res``. Defaults to 0.
            idx_latlon (list, optional): Grid (row, col) index for each entry in
                ``lat_lons``; forwarded to the encoder and decoder. Defaults to None.
            concat_decoder_neighbours (bool, optional): Forwarded to
                ``SatelliteDecoder`` as ``concat_neighbours``. Defaults to False.
            concat_decoder_neighbours_2 (bool, optional): Forwarded to
                ``SatelliteDecoder`` as ``concat_neighbours_2``. Defaults to False.
            scatter (str, optional): Scatter reduction forwarded to
                ``SatelliteProcessor``. Defaults to "mean".
            disaggregated (bool, optional): Forwarded to ``SatelliteProcessor``. Defaults to False.
            batchsize (int, optional): <FILL IN> — accepted but not currently used
                anywhere in this constructor. Defaults to 5.
            attention (bool, optional): Forwarded to the encoder and processor.
                Defaults to False.
            release_coords (str, optional): Forwarded to the encoder. Defaults to "default".
            release_edges (bool, optional): Forwarded to the encoder. Defaults to False.
            decoder_append_latlon (bool, optional): Forwarded to ``SatelliteDecoder``
                as ``append_latlon``. Defaults to False.
            concat_enc_neighbours (bool, optional): Forwarded to the encoder. Defaults to False.
            initial_enc (bool, optional): Forwarded to the encoder. Defaults to False.
            initial_enc_dim (int, optional): Output dimension of the encoder's initial
                encoding MLP (only used when ``initial_enc`` is True); forwarded to the
                encoder. If None, defaults to the encoder's ``output_dim`` (``node_dim``
                here). Defaults to None.
            wind_mesh_edges (bool, optional): Whether to use dynamic wind-based mesh
                edges; forwarded to the encoder. See
                ``gates.training.training.setup_dynamic_edges``. Defaults to False.
            wind_indices (list, optional): Feature indices for wind-based dynamic
                edges; forwarded to the encoder. Defaults to None.
            latlon_mesh_edges (bool, optional): Whether to use dynamic lat/lon-based
                mesh edges; forwarded to the encoder. Defaults to False.
            latlon_indices (list, optional): Feature indices for lat/lon-based
                dynamic edges; forwarded to the encoder. Defaults to None.
            dynamic_earthdistance (bool, optional): Whether to compute dynamic earth
                distance edges; forwarded to the encoder. Defaults to False.
        """
        
        super().__init__()
        #print(f"forecast - dropout {dropout}")
        self.feature_dim = feature_dim
        if output_dim is None:
            output_dim = self.feature_dim
        #print("set up encoder")

        if use_dynamic_encoder:
            self.encoder = SatelliteDynamicEncoder(
                lat_lons=lat_lons,
                whole_world=whole_world,
                resolution=resolution,
                input_dim=feature_dim,
                output_dim=node_dim,
                output_edge_dim=edge_dim,
                hidden_dim_processor_edge=hidden_dim_processor_edge,
                hidden_layers_processor_node=hidden_layers_processor_node,
                hidden_dim_processor_node=hidden_dim_processor_node,
                hidden_layers_processor_edge=hidden_layers_processor_edge,
                mlp_norm_type=norm_type,
                use_checkpointing=use_checkpointing, dropout=dropout,higher_res=higher_mesh_res,idx_latlon=idx_latlon, attention=attention, release_coords=release_coords, release_edges=release_edges, concat_enc_neighbours=concat_enc_neighbours,initial_enc=initial_enc, initial_enc_dim=initial_enc_dim,
                wind_mesh_edges=wind_mesh_edges,
                wind_indices=wind_indices,
                latlon_mesh_edges=latlon_mesh_edges,
                latlon_indices=latlon_indices,
                dynamic_earthdistance=dynamic_earthdistance
            )
        else:
            # Static-encoder A/B test path. Drops the dynamic-edge kwargs and forces
            # better_meshnodes=False so this reduces to the exact default path the
            # dynamic encoder runs (no +2 node features, no idx_latlon requirement).
            if wind_mesh_edges or latlon_mesh_edges or dynamic_earthdistance:
                raise ValueError(
                    "use_dynamic_encoder=False cannot be combined with wind_mesh_edges/"
                    "latlon_mesh_edges/dynamic_earthdistance (SatelliteEncoder has no dynamic edges)."
                )
            self.encoder = SatelliteEncoder(
                lat_lons=lat_lons,
                whole_world=whole_world,
                resolution=resolution,
                input_dim=feature_dim,
                output_dim=node_dim,
                output_edge_dim=edge_dim,
                hidden_dim_processor_edge=hidden_dim_processor_edge,
                hidden_layers_processor_node=hidden_layers_processor_node,
                hidden_dim_processor_node=hidden_dim_processor_node,
                hidden_layers_processor_edge=hidden_layers_processor_edge,
                mlp_norm_type=norm_type,
                use_checkpointing=use_checkpointing, dropout=dropout,higher_res=higher_mesh_res,idx_latlon=idx_latlon, attention=attention, release_coords=release_coords, release_edges=release_edges, concat_enc_neighbours=concat_enc_neighbours,initial_enc=initial_enc, initial_enc_dim=initial_enc_dim,
                better_meshnodes=False
            )
        if not encode_edges:
            edge_dim=2
        if not encode_nodes:
            node_dim=feature_dim
        
        print("set up processor")
        self.processor = SatelliteProcessor(
            input_dim=node_dim,
            edge_dim=edge_dim,
            num_blocks=num_blocks,
            hidden_dim_processor_edge=hidden_dim_processor_edge,
            hidden_layers_processor_node=hidden_layers_processor_node,
            hidden_dim_processor_node=hidden_dim_processor_node,
            hidden_layers_processor_edge=hidden_layers_processor_edge,
            mlp_norm_type=norm_type, dropout=dropout, scatter=scatter, disaggregated=disaggregated, attention=attention, attention_mask=self.encoder.attention_mask, residuals=residuals
        )
        print("set up decoder")

        self.decoder = SatelliteDecoder(
            lat_lons=lat_lons,
            h_grid=self.encoder.h3_grid,
            whole_world=whole_world,
            resolution=resolution,
            input_dim=node_dim,
            output_dim=output_dim,
            mlp_norm_type=norm_type,
            hidden_dim_decoder=hidden_dim_decoder,
            hidden_layers_decoder=hidden_layers_decoder,
            use_checkpointing=use_checkpointing , dropout=dropout, final_activation=decoder_final_layer, n_neighbours=n_decoder_neighbours, concat_neighbours=concat_decoder_neighbours, concat_neighbours_2=concat_decoder_neighbours_2,idx_latlon=idx_latlon, append_latlon=decoder_append_latlon
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run the encode-process-decode pipeline to compute the forecast.

        Args:
            features (torch.Tensor): The input features, aligned with the order of ``lat_lons``.

        Returns:
            torch.Tensor: The forecast output, decoded back onto the lat/lon graph.
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
