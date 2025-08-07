from typing import List

import torch
from torch_scatter import scatter_add

def get_edge_probs_split_layerwise(edge_probs: torch.Tensor, num_nodes_layerwise: torch.Tensor,
                                   edges_per_node: torch.Tensor) -> List[torch.Tensor]:
    index = torch.repeat_interleave(num_nodes_layerwise)
    num_edges_layerwise_sum = scatter_add(edges_per_node, index)[:-1].tolist()
    edge_probs_split_layerwise = torch.split(edge_probs, num_edges_layerwise_sum)
    return edge_probs_split_layerwise, num_edges_layerwise_sum


@torch.jit.script
def get_padded_edge_values(edge_values: torch.Tensor, edge_mask: torch.Tensor, num_edges: int):
    masked_edge_values = edge_values
    edge_values = torch.empty((num_edges, ), device=masked_edge_values.device).fill_(torch.inf)
    edge_values[~edge_mask] = masked_edge_values
    return edge_values

def calculate_weights_for_edge_probs(edge_probs_layer: torch.Tensor):
    inf_mask = (edge_probs_layer == torch.inf)
    weights =  1 - edge_probs_layer # torch.norm((edge_probs_layer[:,None,:] - target_probs_layer[None,:,:]), dim=-1)
    weights[inf_mask] = torch.inf
    return weights.detach().cpu().numpy()