from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.nn import functional as F
from torch_geometric import transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix

from src.utils.transforms.normalize import InvariantNorm
from src.utils.transforms.edge import SphericalInverse

from src.utils.pandas import is_preprocessed, preprocess, remove_disconnected


class HitGraph(Data):
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor, 
                 y: torch.Tensor, is_tracker: torch.Tensor, next_is_tracker: torch.Tensor,
                 edep: torch.Tensor, z: torch.Tensor, skip_tracker: bool, to_wet: bool,
                 is_primary: torch.Tensor, spot_x: float = None, spot_y: float = None,
                 *args, **kwargs) -> None:
        """Use from_csv or from_df for creating a new TGraph object!"""
        super().__init__(x=x, edge_index=edge_index, pos=pos, y=y, edep=edep)
        
        self._spot_x = spot_x
        self._spot_y = spot_y 
        self._edep = edep
        self._to_wet = to_wet
        self._is_primary = is_primary
        self._is_tracker = is_tracker
        self._next_is_tracker = next_is_tracker
        self._z = z
        self._contains_tracker = not skip_tracker
        _, c =self.z.unique(return_counts=True)
        self._layerwise_node_count: torch.Tensor = c.flip(dims=(0, ))
        edge_index_attr = torch.arange(self.edge_index.shape[1])
        self._edge_adjacency_csr = to_scipy_sparse_matrix(self.edge_index, edge_index_attr).tocsr()

        self._epn_zero_cumsum = F.pad(self.edges_per_node, (1, 1), 'constant', 0).cumsum(dim=-1)
        self._edge_labels = (self.y[self.edge_index[0]] == self.y[self.edge_index[1]]).float()
    
    @classmethod
    def from_csv(cls, path: str, skip_tracker: bool = False, to_wet: bool = False, 
                 filter_secondaries: bool = False, cluster_threshold: float = None,
                 device: str = "cpu") -> "HitGraph": 
        """Generate a Hit Graph using a csv file containing all detector information as input.

        Args:
            path (str): Filepath to *.csv file containing the detector readout data.
            skip_tracker (bool, optional): Excluding tracking layers from graph consruction (debugging + testing). Defaults to False.
            to_wet (bool, optional): Determines whether the relative z positions should be converted to water equivalent thicknesses, defaults to False
            filter_secondaries (bool, optional): Exclude secondaries from graph construction (debugging + testing). Defaults to False.
            cluster_threshold (float, optional): Cluster size threshold for secondary hit removal. Defaults to None.
            device (str, optional): Device to create graph on. Defaults to "cpu".

        Returns:
            HitGraph: Hit graph object
        """
        return HitGraph.from_df(pd.read_csv(path), skip_tracker, to_wet, 
                              filter_secondaries, cluster_threshold, device)
    
    @classmethod
    def from_df(cls, df: pd.DataFrame, skip_tracker: bool = False, 
                to_wet: bool = False, filter_secondaries: bool = False,
                cluster_threshold: float = None, device: str = "cpu"):
        """Generate a Hit Graph using a loaded Data Frame all detector information as input.

        Args:
            path (str): Filepath to *.csv file containing the detector readout data.
            skip_tracker (bool, optional): Excluding tracking layers from graph consruction (debugging + testing). Defaults to False.
            to_wet (bool, optional): Determines whether the relative z positions should be converted to water equivalent thicknesses, defaults to False
            filter_secondaries (bool, optional): Exclude secondaries from graph construction (debugging + testing). Defaults to False.
            cluster_threshold (float, optional): Cluster size threshold for secondary hit removal. Defaults to None.
            device (str, optional): Device to create graph on. Defaults to "cpu".

        Returns:
            HitGraph: Hit graph object
        """

        def error_message_col(name: str):
            return f"Dataframe should contain {name} column"

        assert "edep" in df.columns, error_message_col("edep")
        assert "posX" in df.columns, error_message_col("posX")
        assert "posY" in df.columns, error_message_col("posY")
        assert "posZ" in df.columns, error_message_col("posZ") 
        
        # Removes all particle that are not primary particles 
        if filter_secondaries and "parentID" in df.columns:
            df = df[df.parentID == 0]
        
        if not is_preprocessed(df):
            df = preprocess(df)

        if cluster_threshold is not None:
            df = df[df.cluster >= cluster_threshold]
        
        df = remove_disconnected(df)
        
        if skip_tracker:
            df = df[df.z >= 2] 
        
        df_sorted = df.sort_values('z', ascending=False)
        
        edep = cls.extract_edep_df(df_sorted, device)
        y, is_tracker, next_is_tracker, is_primary = cls.extract_ground_truth_df(df_sorted, device)
        x = cls.extract_node_features_df(df_sorted, pixel=False, one_hot_layer=False, device=device)
        z = cls.extract_z_index_df(df_sorted, device=device)
        pos = cls.extract_node_positions_df(df_sorted, pixel=False, device=device)
        edge_index = cls.generate_edge_index(df_sorted, device)
        
        spot_x, spot_y = None, None
        if "spotX" in df_sorted.columns and "spotY" in df_sorted.columns:
            assert len(df_sorted["spotX"].unique()) == 1, "Spot position must be unique."
            assert len(df_sorted["spotY"].unique()) == 1, "Spot position must be unique."
            spot_x = df_sorted["spotX"].iloc[0]
            spot_y = df_sorted["spotY"].iloc[0]
          

        transform = T.Compose([SphericalInverse(norm=False), InvariantNorm()])
        return transform(HitGraph(x=x, edge_index=edge_index, pos=pos, y=y,
                                is_tracker=is_tracker, next_is_tracker=next_is_tracker,
                                edep=edep, z=z, skip_tracker=skip_tracker, to_wet=to_wet, 
                                is_primary=is_primary, spot_x=spot_x, spot_y=spot_y))
    
    @classmethod
    def generate_edge_index(cls, df: pd.DataFrame, device: str) -> torch.Tensor:
        """Generates the edge index for the track data connecting the one-hop neighborhood 
        between two layers (fully connected).

        Args:
            df (pd.DataFrame): Sorted dataframe (ascending, by z) containing all particle hits.
            device (str): Pytorch device used for creating tensors.
        
        Returns:
            torch.Tensor: Generated edge_index tensor [2, n_edge_connections]
        """    
        num_nodes_layerwise = torch.tensor(df.groupby("z").size().values, device=device).flip(dims=(0,))

        repeats_edges = num_nodes_layerwise[1:].repeat_interleave(num_nodes_layerwise[:-1])

        nodes_from_unrepeated = torch.arange(0, num_nodes_layerwise[:-1].sum(), device=device)
        edges_from = nodes_from_unrepeated.repeat_interleave(repeats_edges)

        edges_to = torch.arange(num_nodes_layerwise[0], len(edges_from) + num_nodes_layerwise[0], device=device)

        edge_offset = repeats_edges.cumsum(dim=0).repeat_interleave(repeats_edges) - repeats_edges.repeat_interleave(repeats_edges)
        layerwise_node_offset = F.pad(num_nodes_layerwise[1:], pad=(1, 0))[:-1].cumsum(dim=0)\
                                .repeat_interleave(num_nodes_layerwise[:-1])\
                                .repeat_interleave(repeats_edges) 
        edges_to = edges_to - edge_offset + layerwise_node_offset

        edge_index = torch.stack([edges_from, edges_to])
                
        return edge_index
    
    @classmethod
    def extract_node_features_df(cls, df_sorted: pd.DataFrame, pixel: bool = False,
                                 one_hot_layer: bool = True, device: str = "cpu") -> torch.Tensor:
        edep = cls.extract_cluster_df(df_sorted, device)
        pos = cls.extract_node_positions_df(df_sorted, pixel=pixel, device=device)
        z_indices = cls.extract_z_indices(df_sorted, device)
        return torch.cat([edep.unsqueeze(-1), pos[:,:-1], F.one_hot(z_indices, 50)], dim=1) \
            if one_hot_layer\
            else torch.cat([edep.unsqueeze(-1), pos], dim=1)
    
    @classmethod 
    def extract_z_index_df(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:  
        return torch.tensor(df_sorted.z.values, dtype=torch.float32, device=device)
    
    @classmethod 
    def extract_edep_df(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:  
        return torch.tensor(df_sorted.edep.values, dtype=torch.float32, device=device)

    @classmethod
    def extract_cluster_df(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:
        return torch.tensor(df_sorted.cluster.values, dtype=torch.float32, device=device)

    @classmethod
    def extract_z_indices(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:
        return torch.tensor(df_sorted.z.values, dtype=torch.long, device=device)
    
    @classmethod
    def extract_ground_truth_df(cls, df_sorted: pd.DataFrame, device: str) -> torch.Tensor:
        event_id = None if "eventID" not in df_sorted.columns \
                        else torch.tensor(df_sorted.eventID.values, 
                                          dtype=torch.long, device=device)
                        
        is_primary = None if "parentID" not in df_sorted.columns \
                          else torch.tensor(df_sorted.parentID.values == 0,
                                            device=device)
        
        is_tracker = torch.tensor(df_sorted.z.values < 2, device=device)
        next_is_tracker = torch.tensor(df_sorted.z.values < 3, device=device)
        
        return event_id, is_tracker, next_is_tracker, is_primary
    
    
    def to(self, device: Union[int, str], *args: List[str], non_blocking: bool = False):
        self._edep = self._edep.to(device)
        self._is_tracker = self._is_tracker.to(device)
        self._next_is_tracker = self._next_is_tracker.to(device)
        self._layerwise_node_count = self._layerwise_node_count.to(device)
        self._epn_zero_cumsum = self._epn_zero_cumsum.to(device)
        return super().to(device, *args, non_blocking=non_blocking)
            
    
    @classmethod
    def extract_node_positions_df(cls, df_sorted: pd.DataFrame, pixel: bool = False,
                                  device: str = "str") -> torch.Tensor:
        if pixel:
            return torch.stack([torch.tensor(df_sorted.x.values, dtype=torch.float32, device=device), 
                                torch.tensor(df_sorted.y.values, dtype=torch.float32, device=device), 
                                torch.tensor(df_sorted.z.values, dtype=torch.float32, device=device)]).T
        else:
            return torch.stack([torch.tensor(df_sorted.posX.values, dtype=torch.float32, device=device), 
                                torch.tensor(df_sorted.posY.values, dtype=torch.float32, device=device), 
                                torch.tensor(df_sorted.posZ.values, dtype=torch.float32, device=device)]).T
        
    @property
    def edge_adjacency_csr(self) -> csr_matrix:
        return self._edge_adjacency_csr
    
    @property 
    def edep(self) -> torch.Tensor:
        return self._edep
    
    @property
    def z(self) -> torch.Tensor:
        return self._z
    
    @property
    def is_primary(self) -> torch.Tensor:
        return self._is_primary
    
    @property
    def is_tracker(self) -> torch.Tensor:
        return self._is_tracker
    
    @property
    def next_is_tracker(self) -> torch.Tensor:
        return self._next_is_tracker
    
    @property
    def edge_labels(self):
        return self._edge_labels
    
    @edge_labels.setter
    def edge_labels(self, val):
        self._edge_labels = val
    
    @property
    def num_nodes_layerwise(self):
        return self._layerwise_node_count.to(self.x.device)
    
    @property
    def num_layers(self):
        return len(self._layerwise_node_count)
    
    @property
    def edges_per_layer(self):
        epn = self.edges_per_node
        nnl = self.num_nodes_layerwise.tolist()
        return torch.tensor([x.sum() for x  in torch.split(epn, nnl)])
    
    @property
    def edges_per_node(self):
        epn = torch.zeros_like(self.x[:,0], dtype=torch.long)
        _, x = torch.unique(self.edge_index[0], return_counts=True)
        epn[:len(x)] = x
        return epn
    
    @property
    def epn_zero_cumsum(self):
        return self._epn_zero_cumsum
    
    @property
    def contains_tracker(self):
        return self._contains_tracker    
    
    @property
    def to_wet(self):
        return self._to_wet    
    
    @property
    def spot_x(self):
        return self._spot_x
    
    @property 
    def spot_y(self):
        return self._spot_y
    

class MaskedHitGraph(HitGraph):
    """Subclass of HitGraph that supports edge masking based on angular thresholds.

    This class allows masking of edges in the graph based on angle (theta) between
    hits and a reference direction (usually the beamline). It provides utilities
    to work with masked edge indices, features, and derived metrics. Masks are not 
    permanently applied to simplify analysis.
    """
    def __init__(self, edge_mask: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._edge_mask = edge_mask
        self.recalculate_masked_edge_adjacency_csr()
        
    @classmethod
    def from_graph(cls, G: HitGraph, theta_threshold_detector: float, theta_threshold_tracker: float):
        """Construct a MaskedHitGraph from an existing HitGraph by applying angular masking.

        Args:
            G (HitGraph): The source graph to mask.
            theta_threshold_detector (float): Angular threshold for masking non-tracker edges (in radians).
            theta_threshold_tracker (float): Angular threshold for masking tracker edges (in radians).

        Returns:
            MaskedHitGraph: A new instance with edges masked based on provided thresholds.
        """

        #Calculate masked edges in edge_index -> True should be removed
        nit = G.next_is_tracker[G.edge_index[0]]
        up_vector = torch.tensor([0, 0, 1], device=G.x.device)
        dist = G.pos[G.edge_index[0]] - G.pos[G.edge_index[1]]
        thetas = torch.acos(torch.cosine_similarity(dist, up_vector, dim=-1))
        thetas = torch.nan_to_num(thetas, 0.0) #0 if angle is nan

        edge_mask = (((thetas >= theta_threshold_detector) & ~nit) | 
                     ((thetas >= theta_threshold_tracker) & nit))
        
        MG = MaskedHitGraph(**G.__dict__["_store"], is_tracker=G.is_tracker, 
                            next_is_tracker=G.next_is_tracker, z=G.z, 
                            skip_tracker=not G.contains_tracker, to_wet=G.to_wet, 
                            is_primary=G.is_primary, edge_mask=edge_mask)
        
        # Is calculated from scratch for TGraph
        MG.edge_attr = G.edge_attr
        return MG
    
    def recalculate_masked_edge_adjacency_csr(self):
        ones = torch.ones(self.masked_edge_index.shape[1])
        self._masked_edge_adjacency_csr = to_scipy_sparse_matrix(self.masked_edge_index, ones).tocsr()

    def to(self, device: Union[int, str], *args: List[str], non_blocking: bool = False):
        self._edge_mask = self._edge_mask.to(device)
        self._layerwise_node_count = self._layerwise_node_count.to(device)
        
        return super().to(device, *args, non_blocking=non_blocking)
    
    @property
    def edge_mask(self) -> torch.Tensor:
        return self._edge_mask
    
    @edge_mask.setter
    def edge_mask(self, x: torch.Tensor) -> None:
        self._edge_mask = x
        self.recalculate_masked_edge_adjacency_csr()

    @property
    def num_edges_masked(self) -> int:
        return self.masked_edge_index.shape[1]

    @property
    def masked_edges_per_node(self):
        epn = torch.zeros_like(self.x[:,0], dtype=torch.long)
        _, x = torch.unique(self.masked_edge_index[0], return_counts=True)
        epn[:len(x)] = x
        return epn

    @property
    def masked_edges_per_layer(self):
        epn = self.masked_edges_per_node
        nnl = self.num_nodes_layerwise.tolist()
        return torch.tensor([x.sum() for x  in torch.split(epn, nnl)])

    @property
    def masked_edge_labels(self):
        return self.edge_labels[~self.edge_mask]
        
    @property
    def masked_edge_index(self) -> torch.Tensor:
        return self.edge_index[:,~self.edge_mask]
    
    @property
    def masked_edge_attr(self) -> torch.Tensor:
        return self.edge_attr[~self.edge_mask]
    
    @property
    def masked_edge_adjacency_csr(self):
        return self._masked_edge_adjacency_csr