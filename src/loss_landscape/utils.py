from typing import List, Union, Generator
import os

import torch
import numpy as np
from torch import nn
from glob import glob
from sklearn.decomposition import PCA
from src.utils.experiments import ExperimentBase
from src.utils.experiments import set_random_seeds
from src.loss_landscape.hessian.hessian import Hessian


def load_parameter_matrix_from_checkpoints(exp: ExperimentBase, run: int, device: str, 
                                           early_stopping: bool = True) -> torch.Tensor:
    
    matrix_path = os.path.join(exp.model.checkpoints.checkpoint_dir, 
                               f"run_{run}"
                               f"param_matrix_{early_stopping}.npy")

    if not os.path.exists(matrix_path):
        parameter_matrix = []

        for model in exp.model.iterate_checkpoints(run=run, device=device, early_stopping=early_stopping):
            parameter_matrix += [torch.cat([p.detach().cpu().flatten() for p in model.parameters()])]
            del model

        matrix = torch.stack(parameter_matrix)
        np.save(matrix_path, matrix.cpu().numpy())
    else:
        matrix = np.load(matrix_path)
        matrix = torch.from_numpy(matrix)

    return matrix


def parameters_to_vector(parameters):
    return torch.cat([p.flatten() for p in parameters])


def vector_to_parameter_shape(parameters: Union[List[torch.nn.Parameter], Generator], 
                              parameter_like_vector: torch.Tensor) -> List[torch.Tensor]:

    if isinstance(parameters, Generator):
        parameters = list(parameters)
        
    shapes = [p.shape for p in parameters]
    sizes = [p.numel() for p in parameters]
    parameter_shape_list = list(parameter_like_vector.split(sizes))

    for i in range(len(parameter_shape_list)):
        parameter_shape_list[i] = parameter_shape_list[i].reshape(shapes[i])

    return parameter_shape_list


def get_principal_components_for_parameter_matrix(parameter_matrix: torch.Tensor):
    pca = PCA(2).fit(parameter_matrix)
    v0 = torch.from_numpy(pca.components_[0])
    v1 = torch.from_numpy(pca.components_[1])

    evr = pca.explained_variance_ratio_

    return (v0, v1), (evr[0], evr[1])


def get_hessian_eigenvectors(exp: ExperimentBase, run: int, device: str, 
                             max_iter: int = 100, tol: float = 1e-3,
                             early_stopping: bool = True):
    
    dataset = exp.dataset.load_train_from_config(device)
    transforms = exp.transforms.generate_from_config()
    solver = None if exp.solver == None else exp.solver.generate_from_config()

    set_random_seeds(0) #Allows for reproducible hessian eigenvectors
    model = exp.model.load_from_config(run, device, early_stopping=early_stopping)
    hessian = Hessian(model, transforms, solver, dataset, device)
    _, eigenvectors = hessian.eigenvalues(top_n=2, maxIter=max_iter, tol=tol)
    v0 = parameters_to_vector(eigenvectors[0]).to("cpu")
    v1 = parameters_to_vector(eigenvectors[1]).to("cpu")

    return v0, v1
