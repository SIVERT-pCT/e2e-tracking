from itertools import compress
from math import pi as PI
from typing import Optional, Union, List, Generator

import torch
import numpy as np
import pandas as pd
from torch import nn
import networkx as nx
from torch.nn import functional as F
from torch_geometric.utils import to_networkx

from src.supervised.gnn.models import HGInteractionNetwork, LGInteractionNetwork
from src.utils.dataset import HitGraphDataset
from src.utils.graph import HitGraph, MaskedHitGraph
from src.utils.pandas import preprocess
from src.utils.transforms.graph import CustomGraphTransform
from src.supervised.combinatorial.lsa import CombinatorialSolver


def subset(dataset: HitGraphDataset, indices: np.ndarray) -> Generator[HitGraph, torch.Any, None]:
    """Helper function for restricting a full HitGraphDataset to a smaller subset of elements defined by a list of dataset indices.

    Args:
        dataset (HitGraphDataset): Original hit-graph dataset, which should be reduced to a subset of elements defined by indices.
        indices (np.ndarray): Numpy array containing a subset of selected indices.

    Yields:
        Generator[HitGraph, torch.Any, None]: Subset of dataset, yielding graph data at indices.
    """
    for index in indices:
        yield dataset.load_graph(index)

def get_metrics_for_dataset(dataset: HitGraphDataset, transforms: CustomGraphTransform, 
                            network: Union[HGInteractionNetwork, LGInteractionNetwork], solver: CombinatorialSolver, 
                            events: int, indices: Optional[np.ndarray] = None) -> pd.DataFrame:
    """_summary_

    Args:
        dataset (HitGraphDataset): HitGraph dataset used that should be used for evaluation.
        transforms (CustomGraphTransform): Sequence of graph transforms used for preprocessing the initial hit-graph data.
        network (Union[HGInteractionNetwork, LGInteractionNetwork]): Optimized interaction network architecture (either HGInteractionNetwork or LGInteractionNetwork).
        solver (CombinatorialSolver): Instance of CombinatorialSolver, used for creating unique assignment of graph edges (both for PTT and PAT models).
        events (int): Number of particl tracks of events in a single readout frame.
        indices (Optional[np.ndarray], optional): Subset of indices to calculate results for smaller sample sizes (e.g., during training). Defaults to None.

    Returns:
        pd.DataFrame: Dataframe containing a list of spot positions and reconstruction purities and efficiencies for every readout frame. 
    """
    
    assert transforms.to_line_graph == isinstance(network, LGInteractionNetwork), "Invalid configuration. Transform must match model config!"
    
    if solver ==None: solver = CombinatorialSolver(1.0, 0.0, False)
    ds_enumerable = dataset if indices is None else subset(dataset, indices)

    with torch.inference_mode():
        results = []

        for G in ds_enumerable:
            HG, LG = transforms(G)
            y = HG.masked_edge_labels
            edge_logits = network(LG) if LG != None else network(HG)
            edge_probs = torch.sigmoid(edge_logits)
            edge_probs = solver(HG, edge_probs, train=True)
            mask = edge_probs > 0.5

            # Create compressed graph representation with only the final
            # edges, generating the predicted particle tracks after unique assignment.
            edge_index_masked = HG.masked_edge_index[:,mask]
            HG.edge_index = edge_index_masked
            HG.edge_index = HG.edge_index[[1, 0]]

            # Find all tracks by finding connected paths from root to leave nodes
            # Root nodes: No incoming edges (end of track, last layer)
            # Leaf nodes: No outgoing edges (end of track, first layer)
            nx_graph = to_networkx(HG)
            roots  = [v for v, d in nx_graph.in_degree() if d == 0]
            leaves = [v for v, d in nx_graph.out_degree() if d == 0]
            tracks = []

            for root in roots:
                paths = nx.all_simple_paths(nx_graph, root, leaves)
                tracks.extend(paths)
            
            #Perform track filtering based on track length and energy cuts in last layer
            tracks = [t for t in tracks if len(t) > 4 and (HG.edep[t[-1]] > 0.0625)] 

            pur = purity(G, tracks)
            eff = efficiency(G, true_primary_tracks_from_graph(G), tracks)
            results.append((G.spot_x, G.spot_y, pur, eff))
            
        results = pd.DataFrame(results)
        results.columns = ["spotX", "spotY", "pur", "eff"]
        
        return results


def true_tracks_from_graph(graph: Union[HitGraph, MaskedHitGraph]) -> List[List[int]]:
    """ Returns for any HitGraph or MaskedHitGraph a nested list containing all particle tracks in the readout frame. 

    Args:
        graph (Union[HitGraph, MaskedHitGraph]): Single readout frame, represented as either as a HitGraph or MaskedHitGraph.

    Returns:
        List[List[int]]: Nested list of integers, containing all particle tracks, each defined by the corresponding hit-graph node indices.
    """
    node_idx = torch.arange(graph.y.shape[0], device=graph.y.device)
    s, indices = torch.sort(graph.y)
    _, counts = torch.unique_consecutive(s, return_counts=True)

    true_tracks = [sorted(t.tolist(), reverse=True) for t in torch.split(node_idx[indices], counts.tolist())] 
    return true_tracks


def true_primary_tracks_from_graph(graph: Union[HitGraph, MaskedHitGraph]) -> List[List[int]]:   
    """ Returns for any HitGraph or MaskedHitGraph a nested list containing all primary particle tracks
     (all hits belong to a primary particle) in the readout frame. 

    Args:
        graph (Union[HitGraph, MaskedHitGraph]): Single readout frame, represented as either as a HitGraph or MaskedHitGraph.

    Returns:
        List[List[int]]: Nested list of integers, containing all particle tracks, each defined by the corresponding hit-graph node indices.
    """
    node_idx = torch.arange(graph.y.shape[0], device=graph.y.device)
    s, indices = torch.sort(graph.y)
    _, counts = torch.unique_consecutive(s, return_counts=True)

    true_tracks = [sorted(t.tolist(), reverse=True) \
        for t in torch.split(node_idx[indices], counts.tolist()) \
        if torch.all(graph.is_primary[t])] 
    
    return true_tracks


def purity(graph: Union[HitGraph, MaskedHitGraph], tracks) -> float:
    corr = [(graph.y[t] == graph.y[t[0]]).all() for t in tracks if graph.is_primary[t].all()] 
    if len(corr) == 0: return 0
    return (torch.stack(corr).sum()/len(corr)).cpu().numpy()


def efficiency(graph: Union[HitGraph, MaskedHitGraph], true_sets, tracks) -> float:
    corr = [(graph.y[t] == graph.y[t[0]]).all() for t in tracks if graph.is_primary[t].all()]
    if len(corr) == 0: return 0
    return (torch.stack(corr).sum()/len(true_sets)).cpu().numpy()

