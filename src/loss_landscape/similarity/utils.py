import torch
from torch import nn
from src.supervised.gnn.layers import RelationalModel, ObjectModel

def iterate_minibatches(x: torch.Tensor, y: torch.Tensor, n: int = 4048, shuffle: bool = False):
    assert x.shape[0] == y.shape[0]
    N = x.shape[0]
    indices = torch.randperm(N) if shuffle else torch.arange(N)
    n_minibatches = N // n
    for i in range(n_minibatches):
        i_minibatch = indices[i*n:(i+1)*n]
        yield x[i_minibatch], y[i_minibatch]


def hisc(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        N = K.shape[0]
        ones = torch.ones((N, 1)).to(K.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result)

    
def generate_gram_matrix(X: torch.Tensor) -> torch.Tensor:
    K = X @ X.T
    K.fill_diagonal_(0.0)
    return K


def is_relevant_layer(layer) -> bool:
     return (not isinstance(layer, nn.Sequential) and 
             not isinstance(layer, RelationalModel) and 
             not isinstance(layer, ObjectModel))