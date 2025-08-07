from ast import Tuple
import ray
import torch
from torch import nn
from typing import List, Tuple

from src.utils.experiments import ExperimentBase
from src.supervised.utils.losses import hamming_loss
from src.loss_landscape.coordinates import Coordinates

def evaluate_loss(model: nn.Module, exp: ExperimentBase, device: str):
    solver = exp.solver.generate_from_config()
    transforms = exp.transforms.generate_from_config()
    dataset = exp.dataset.load_train_from_config(device)
    
    with torch.no_grad():
        loss = 0
        for G in dataset:
            HG, LG = transforms(G)

            edge_probs = solver(HG, torch.sigmoid(model(LG)), 1.0)
            edge_labels = HG.masked_edge_labels

            loss += hamming_loss(edge_probs, edge_labels)
    
    return loss/len(dataset)

@ray.remote(num_gpus=0.3, num_cpus=5)
def evaluate_loss_for_parameters(exp: ExperimentBase, coordinates: Coordinates, a: float, b: float):
    device = "cuda"
    # Load to cpu first since coordinates is a cpu operation 
    model = exp.model.generate_from_config("cpu")
    model.load_state_dict(coordinates(a, b))
    model = model.to(device)

    return evaluate_loss(model, exp, device)


@ray.remote(num_gpus=0.3, num_cpus=5)
def evaluate_loss_for_parameter_chunk(exp: ExperimentBase, coordinates: Coordinates, param_chunk: List[Tuple[float, float]]):
    losses = []
    device = "cuda"
    model = exp.model.generate_from_config("cpu")

    for a, b in param_chunk:
        model.to("cpu")
        model.load_state_dict(coordinates(a, b))
        model = model.to(device)

        losses += [evaluate_loss(model, exp, device)]

    return losses