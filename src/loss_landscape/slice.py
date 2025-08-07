import gc
import os
import ray
import ray
import torch
import numpy as np
from torch import nn
from ast import Tuple
from typing import Optional
from typing import List, Tuple
from torch.nn import functional as F

from src.supervised.utils.losses import hamming_loss
from src.utils.experiments import ExperimentBase
from src.loss_landscape.coordinates import Coordinates
from src.utils.experiments import ExperimentBase
from src.utils.eval import subset
from src.supervised.combinatorial.lsa import CombinatorialSolver
from src.loss_landscape.utils import load_parameter_matrix_from_checkpoints, get_hessian_eigenvectors

import warnings
warnings.filterwarnings("ignore")

def evaluate_loss(model: nn.Module, exp: ExperimentBase, device: str, 
                  loss_type: str, evaluate_n: Optional[int] = None):
    
    solver = CombinatorialSolver(1, 1, False) #Arbitrary values since no backward pass
    transforms = exp.transforms.generate_from_config()
    dataset = exp.dataset.load_train_from_config(device)

    if evaluate_n != None:
        indices = np.arange(len(dataset))
        indices = indices[::int(np.ceil( len(indices) / evaluate_n ))]

    ds_enumerable = dataset if indices is None else subset(dataset, indices)
    
    with torch.inference_mode():
        loss = 0
        for G in ds_enumerable:
            HG, LG = transforms(G)
            edge_probs = torch.sigmoid(model(LG))

            if loss_type == "hamming":
                edge_probs = solver(HG, edge_probs, False)

                nit = HG.next_is_tracker[HG.masked_edge_index].any(dim=0)
                loss1 = hamming_loss(edge_probs[nit], HG.masked_edge_labels[nit])
                loss2 = hamming_loss(edge_probs[~nit], HG.masked_edge_labels[~nit])
                loss += (loss1 + loss2)
            elif loss_type =="bce":
                nit = HG.next_is_tracker[HG.masked_edge_index].any(dim=0)
                loss1 = F.binary_cross_entropy(edge_probs[nit], HG.masked_edge_labels[nit])
                loss2 = F.binary_cross_entropy(edge_probs[~nit], HG.masked_edge_labels[~nit])
                loss = (loss1 + loss2)
            else: 
                raise Exception(f"Invalid loss type: {loss_type}")
    
    return loss/len(dataset)

@ray.remote(num_gpus=0.2, num_cpus=5)
def evaluate_loss_for_parameters(exp: ExperimentBase, coordinates: Coordinates, a: float, b: float,
                                 loss_type: str, evaluate_n: Optional[int] = None):
    device = "cuda"
    # Load to cpu first since coordinates is a cpu operation 
    model = exp.model.generate_from_config("cpu")
    model.load_state_dict(coordinates(a, b))
    model = model.to(device)

    return evaluate_loss(model, exp, device, loss_type, evaluate_n)

def generate_loss_lanscape(exp: ExperimentBase, run: int, device: str, 
                           num_steps: int,
                           loss_type: str = "hamming", #One of [hamming, bce]
                           x_range: Tuple[float, float] = (0.5, 0.5),
                           y_range: Tuple[float, float] = (0.5, 0.5),
                           ray_visible_devices: Optional[List[int]] = None,
                           include_ray_dashboard: bool = False,
                           eigenvalue_max_iter: int = 100,
                           eigenvalue_tol: float = 1e-3,
                           force_recompute: bool = False,
                           early_stopping: bool = True,
                           evaluate_n: Optional[int] = None):

    save_dir = os.path.join(exp.model.model_dir, "landscape", f"run_{run}")
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)

    precomputed = True
    for file in ["v0.npy", "v1.npy"]:
        if not os.path.isfile(os.path.join(save_dir, file)):
            precomputed = False
    
    if not precomputed or force_recompute:
        print("Computing hessian eigenvectors with largest eigenvalue ...")
        v0, v1 = get_hessian_eigenvectors(exp, run, device, 
                                          max_iter=eigenvalue_max_iter, 
                                          tol=eigenvalue_tol,
                                          early_stopping=early_stopping)
        v0 = v0.detach()
        v1 = v1.detach()
        torch.cuda.empty_cache()
        gc.collect()
        
        np.save(os.path.join(save_dir, "v0.npy"), v0)
        np.save(os.path.join(save_dir, "v1.npy"), v1)
    else:
        v0 = torch.tensor(np.load(os.path.join(save_dir, "v0.npy")))
        v1 = torch.tensor(np.load(os.path.join(save_dir, "v1.npy")))


    precomputed = os.path.isfile(os.path.join(save_dir, f"loss_landscape_{loss_type}.npy"))

    if not precomputed or force_recompute:
        try:
            if ray_visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"]=",".join(map(str, ray_visible_devices))

            num_gpus = len(ray_visible_devices) \
                if ray_visible_devices != None \
                else torch.cuda.device_count()

            os.environ['PYTHONPATH'] = os.getcwd()  
            ray.init(num_gpus=num_gpus, include_dashboard=include_ray_dashboard)
            
            coordinates = Coordinates(exp.model.load_from_config(run, "cpu"), v0, v1)

            x = np.linspace(x_range[0], x_range[1], num_steps)
            y = np.linspace(y_range[0], y_range[1], num_steps)
            XX, YY = np.meshgrid(x, y)

            print("Generating loss landscape ...")
            losses = ray.get([evaluate_loss_for_parameters.remote(exp, coordinates, x, y, loss_type, evaluate_n) \
                            for x, y in zip(XX.flatten(), YY.flatten())])

            losses = torch.stack(losses).reshape(num_steps, num_steps).detach().cpu().numpy()
            np.save(os.path.join(save_dir, f"loss_landscape_{loss_type}.npy"), losses)
            np.save(os.path.join(save_dir, "xx.npy"), XX)
            np.save(os.path.join(save_dir, "yy.npy"), YY)
            
            if ray_visible_devices:
                del os.environ["CUDA_VISIBLE_DEVICES"]
                del os.environ['PYTHONPATH']
        except Exception:
            raise
        finally:
            try:
                del os.environ["CUDA_VISIBLE_DEVICES"]
                del os.environ['PYTHONPATH']
            except:
                pass
            ray.shutdown()
            
    else: 
        losses = np.load(os.path.join(save_dir, f"loss_landscape_{loss_type}.npy"))
        XX = np.load(os.path.join(save_dir, "xx.npy"))
        YY = np.load(os.path.join(save_dir, "yy.npy"))

    return v0, v1, XX, YY, losses