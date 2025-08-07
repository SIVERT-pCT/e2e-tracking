from typing import Union
from src.utils.graph import HitGraph, MaskedHitGraph

def cost_margin(G: Union[HitGraph, MaskedHitGraph], alpha: float, secondary_sensitive: bool = True):
    
    edge_labels = G.masked_edge_labels if isinstance(G, MaskedHitGraph) else G.edge_labels
    edge_index = G.masked_edge_index if isinstance(G, MaskedHitGraph) else G.edge_index

    margin = ((1 - edge_labels) * alpha/2 - edge_labels * alpha/2)
    return margin * G.is_primary[edge_index].all(dim=0) \
            if secondary_sensitive \
            else margin