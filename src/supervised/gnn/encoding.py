import math
import torch
from torch.nn import functional as F
from torch_geometric import transforms as T

from typing import Union
from src.utils.graph import HitGraph, MaskedHitGraph

class SimilarityPE(T.BaseTransform):
    def __init__(self, d: int, scaling: float) -> None:
        super().__init__()
        self.d = d
        self.div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(scaling) / d))

    def __call__(self, theta):
        self.div_term = self.div_term.to(theta.device)
        pe = torch.empty((theta.shape[0], self.d), device=theta.device)
        pe[..., 0::2] = torch.sin(torch.sqrt(theta[...,None]) * self.div_term[None, None, :])
        pe[..., 1::2] = torch.cos(torch.sqrt(theta[...,None]) * self.div_term[None, None, :])
        return pe