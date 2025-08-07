from typing import Optional
from copy import deepcopy

import torch
from torch import nn

from src.loss_landscape.utils import vector_to_parameter_shape, parameters_to_vector


def normalize_weights(weights, origin):
    return [torch.nan_to_num(w * torch.norm(wo)/torch.norm(w)) for w, wo in zip(weights, origin)]

def random_weights(origin):
    return [torch.randn_like(w) for w in origin]

class Coordinates:
    def __init__(self, model: nn.Module(), 
                 v0_vec: Optional[torch.Tensor] = None, 
                 v1_vec: Optional[torch.Tensor] = None) -> None:
        
        self.origin = deepcopy(list(model.parameters()))
        self.parameter_names = [n for n,_ in model.named_parameters()]
        self.v0 = vector_to_parameter_shape(self.origin, v0_vec.float()) if v0_vec != None else random_weights(self.origin)
        self.v0 = normalize_weights(self.v0, self.origin)
        self.v1 = vector_to_parameter_shape(self.origin, v1_vec.float()) if v0_vec != None else random_weights(self.origin)
        self.v1 = normalize_weights(self.v1, self.origin)

        self.components = torch.stack((parameters_to_vector(self.v0), parameters_to_vector(self.v1)), dim=-1)
        self.origin_vec = parameters_to_vector(self.origin)

        self.bn_parameter_state_dict = dict([(n, p) for n, p in model.state_dict().items() \
                                             if n.split(".")[-1] in ["running_mean", "running_var"]])


    def path_to_coordinates(self, parameter_matrix: torch.Tensor):
        lstsq =  torch.linalg.lstsq(self.components, (parameter_matrix - self.origin_vec).T)
        return lstsq.solution.detach().cpu().numpy()

    def __call__(self, a: float, b: float):
        parameters = [a * w0 + b * w1 + wo
                      for w0, w1, wo in zip(self.v0, self.v1, self.origin)]
        
        parameter_state_dict = dict(zip(self.parameter_names, parameters))
        return {**parameter_state_dict, **self.bn_parameter_state_dict}