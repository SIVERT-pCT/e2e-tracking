import torch
from typing import Optional

def hamming_loss(edge_probs: torch.Tensor, edge_labels: torch.Tensor, weights: Optional[torch.Tensor] = None):
    errors = edge_probs * (1.0 -edge_labels) + (1.0 - edge_probs) * edge_labels

    if weights != None:
        errors = errors * weights

    return errors.mean()

