import torch
import numpy as np
from typing import Union
from lapsolver import solve_dense

from torch_scatter import scatter_add
from torch_scatter.composite import scatter_softmax

from src.utils.graph import HitGraph, MaskedHitGraph
from src.supervised.combinatorial.margins import cost_margin

from src.supervised.combinatorial.utils import get_padded_edge_values, calculate_weights_for_edge_probs


class LSACombinatorialLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge_probs: torch.Tensor, G: Union[HitGraph, MaskedHitGraph], lambda_val: float):
        edge_probs = edge_probs.detach()
        if isinstance(G, MaskedHitGraph):
            edge_probs = get_padded_edge_values(edge_probs, G.edge_mask, G.num_edges)
            
        index = torch.repeat_interleave(G.num_nodes_layerwise)
        num_edges_layerwise_sum = scatter_add(G.edges_per_node, index)[:-1].tolist()
        edge_probs_split_layerwise = torch.split(edge_probs, num_edges_layerwise_sum)
        
        ctx.weights = []
        ctx.matchings = []
        ctx.num_edges_layerwise_sum = num_edges_layerwise_sum
        ctx.lambda_val = lambda_val
        ctx.G = G
        
        for i in range(len(edge_probs_split_layerwise)):
            edge_probs_layer = edge_probs_split_layerwise[i].reshape(G.num_nodes_layerwise[i], -1)
            weights = calculate_weights_for_edge_probs(edge_probs_layer)
            indices = solve_dense(weights)

            matchings = torch.zeros_like(edge_probs_layer)
            matchings[indices] = 1.

            ctx.weights += [weights]
            ctx.matchings += [matchings]

            
        assignments = torch.cat([a.flatten() for a in ctx.matchings])

        return assignments[~G.edge_mask] if isinstance(G, MaskedHitGraph) else assignments
    
                 
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor):
        if isinstance(ctx.G, MaskedHitGraph):
            grad_outputs = get_padded_edge_values(grad_outputs, ctx.G.edge_mask, ctx.G.num_edges)
        
        grad_outputs_split_layerwise = torch.split(grad_outputs, ctx.num_edges_layerwise_sum)
        gradients = []
        
        for i in range(len(grad_outputs_split_layerwise)):
            grad_outputs_layer = grad_outputs_split_layerwise[i].reshape(ctx.G.num_nodes_layerwise[i], -1)
            
            weights_prime = np.maximum(ctx.weights[i] + ctx.lambda_val * grad_outputs_layer.cpu().numpy(), 0.0)
            indices = solve_dense(weights_prime)
            
            better_matchings = torch.zeros_like(ctx.matchings[i])
            better_matchings[indices] = 1.
            
            gradients += [(ctx.matchings[i] - better_matchings)]
        
        gradient = torch.cat([g.flatten() for g in gradients])
        gradient =  gradient[~ctx.G.edge_mask] \
                        if isinstance(ctx.G, MaskedHitGraph) \
                        else gradient
                        
        return gradient, None, None, None


class CombinatorialSolver:
    def __init__(self, solver_lambda: float, margin_alpha: float, secondary_sensitive: bool) -> None:
        self.solver_lambda = solver_lambda
        self.margin_alpha = margin_alpha
        self.secondary_sensitive = secondary_sensitive

    def __call__(self, G: Union[HitGraph, MaskedHitGraph], edge_probs: torch.Tensor, 
                 train: bool = False) -> torch.Tensor:

        if self.margin_alpha != None and train == True: 
            edge_probs = edge_probs + cost_margin(G, self.margin_alpha, self.secondary_sensitive)
    
        return LSACombinatorialLayer().apply(edge_probs, G, self.solver_lambda)