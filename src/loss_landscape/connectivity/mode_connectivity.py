import gc
import math
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from random import uniform
import torch.nn.functional as F
from scipy.special import binom
from torch.nn import Module, Parameter
from typing import Any, List, Callable, Optional, Tuple

from src.supervised.combinatorial.lsa import CombinatorialSolver
from src.utils import transforms
from src.supervised.utils.losses import hamming_loss
from src.utils.dataset import HitGraphDataset
from src.utils.eval import subset
from src.utils.experiments import ExperimentBase, set_random_seeds
from src.loss_landscape.utils import parameters_to_vector, vector_to_parameter_shape
from src.utils.transforms.graph import CustomGraphTransform


class Bezier(Module):
    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.num_bends = num_bends
        self.register_buffer(
            'binom',
            torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32))
        )
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * \
               torch.pow(t, self.range) * \
               torch.pow((1.0 - t), self.rev_range)


class CurveCoordinates:
    def __init__(self, curve: Bezier, a0: nn.Module, a2: nn.Module, train_set: HitGraphDataset, test_sets: List[HitGraphDataset], 
                 transforms: CustomGraphTransform, solver: Optional[CombinatorialSolver], subset_n_train: int = 100, device: str = "cpu") -> None:
        
        assert curve.num_bends == 3
        self.curve = curve
        self.device = device

        self.transforms = transforms
        self.train_set = train_set
        self.test_sets = test_sets
        self.solver = solver

        self.base_model = a0
        self.parameter_names = [n for n,_ in self.base_model.named_parameters()]
        
        self.anchor_0 = parameters_to_vector(a0.parameters()).detach()
        self.anchor_0.requires_grad = False
        self.anchor_2 = parameters_to_vector(a2.parameters()).detach()
        self.anchor_2.requires_grad = False

        self.indices = np.arange(len(self.train_set))
        self.indices = self.indices[::int(np.ceil( len(self.indices) / subset_n_train))]

        self.anchor_1 = nn.Parameter(self.__init_linear(), requires_grad=True)
        self.parameter_vectors = [self.anchor_0, self.anchor_1, self.anchor_2]

        self.bn_param_state_dict = dict([(n, p) for n, p \
                                         in self.base_model.state_dict().items() \
                                         if self.__is_bn_param(n)])
        
        gc.collect()
        torch.cuda.empty_cache() 


    @staticmethod
    def from_custom(exp_a: ExperimentBase, exp_b: ExperimentBase, run_a: int, run_b: int, device: str, 
                    lambda_val: float = 10., use_bce_loss: bool = False):
        
        if use_bce_loss:
            exp_a.solver = None
            exp_b.solver = None

        if use_bce_loss:
            assert exp_a.solver == None and exp_b.solver == None, \
                "BCE loss is only valid for configurations where both " + \
                "experiments were trained without the LSA solver"

        a0 = exp_a.model.load_from_config(run_a, device)
        a2 = exp_b.model.load_from_config(run_b, device)
        train_set = exp_a.dataset.load_train_from_config(device)
        test_sets = [exp_a.dataset.load_test_from_config(n, 100, device) for n in exp_a.dataset.test_files.keys()]
        transforms = exp_a.transforms.generate_from_config()
        solver = None if use_bce_loss else CombinatorialSolver(lambda_val, 0.0, False)

        return CurveCoordinates(Bezier(3), a0, a2, 
                                train_set, test_sets, 
                                transforms, solver, device=device)

    @staticmethod
    def from_runs(exp: ExperimentBase, run_a: int, run_b: int, device: str):
        return CurveCoordinates.from_custom(exp, exp, run_a, run_b, device)
    
    @staticmethod
    def from_experiments(exp_a: ExperimentBase, exp_b: ExperimentBase, run: int, device: str):
        return CurveCoordinates.from_custom(exp_a, exp_b, run, run, device) 


    def __is_bn_param(self, name: str):
        return (name.split(".")[-1] in ["running_mean", "running_var"])
    
    def __is_var_param(self, name: str):
        return name.split(".")[-1] == "running_var"
    
    def __is_mean_param(self, name: str):
        return name.split(".")[-1] == "running_mean"
    
    def __set_bn_param(self, n, p):
        return (n, torch.zeros_like(p)) if self.__is_mean_param(n) \
          else (n, torch.ones_like(p))
    
    def __init_linear(self, alpha: float = 0.5):
        anchor_1 = (self.anchor_0 * alpha + (1- alpha) * self.anchor_2)
        anchor_1.requires_grad = True
        return anchor_1

    def __call__(self, t: float) -> Any:
        coeffs_t = self.curve(t)
        param_vec_coord = torch.zeros_like(self.parameter_vectors[0])

        for i, coeff in enumerate(coeffs_t):
            param_vec_coord += self.parameter_vectors[i] * coeff

        l2 = torch.sum(param_vec_coord[i] ** 2)

        param_coord = vector_to_parameter_shape(self.base_model.parameters(), param_vec_coord)
        param_state_dict = dict(zip(self.parameter_names, param_coord))
        param_state_dict = {**param_state_dict, **self.bn_param_state_dict}
        
        return param_state_dict, l2
    
    
    def fit(self, n_steps: int = 2000, lr: float = 1e-3, batch_size: int = 8):
        set_random_seeds(0)
        optimizer = torch.optim.RMSprop([self.anchor_1], lr=lr)

        for _ in tqdm(range(n_steps)):
            optimizer.zero_grad()
    
            for _ in range(batch_size):
                state_dict, l2 = self(uniform(0.0, 1.0))
                G = self.train_set.sample()
                HG, LG = self.transforms(G)

                nit = HG.next_is_tracker[HG.masked_edge_index].any(dim=0)

                edge_logits = torch.nn.utils.stateless.functional_call(self.base_model, state_dict, LG)
                edge_probs = torch.sigmoid(edge_logits)

                if self.solver != None:
                    edge_probs = self.solver(HG, edge_probs, train=True)
                    loss1 = hamming_loss(edge_probs[nit], HG.masked_edge_labels[nit])
                    loss2 = hamming_loss(edge_probs[~nit], HG.masked_edge_labels[~nit])
                else:
                    loss1 = F.binary_cross_entropy(edge_probs[nit], HG.masked_edge_labels[nit])
                    loss2 = F.binary_cross_entropy(edge_probs[~nit], HG.masked_edge_labels[~nit])

                loss = (loss1 + loss2)/batch_size
                
                loss.backward()

            optimizer.step()
    
    def eval(self, T: int = 100, on_test_set: bool = False) -> Tuple[np.ndarray, np.ndarray,
                                                                    np.ndarray, np.ndarray,
                                                                    np.ndarray]:
        losses = [];  bce_losses = []; tprs = []; fprs = []
        ts = np.linspace(0.0, 1.0, T)

        solver = self.solver
        if solver == None:
            solver = CombinatorialSolver(1.0, 0.0, False)

        for t in tqdm(ts):
            with torch.inference_mode():
                state_dict, l2 = self(t)
                bce_loss = 0; loss = 0; TPs = []; TNs = []; FPs = []; FNs = []
                datasets = self.test_sets if on_test_set else [subset(self.train_set, indices=self.indices)]
                num_graphs = 0
                for dataset in datasets:
                    for G in dataset:
                        HG, LG = self.transforms(G)
                        nit = HG.next_is_tracker[HG.masked_edge_index].any(dim=0)

                        edge_logits = torch.nn.utils.stateless.functional_call(self.base_model, state_dict, LG)
                        edge_probs = torch.sigmoid(edge_logits)

                        bce_loss1 = F.binary_cross_entropy(edge_probs[nit], HG.masked_edge_labels[nit])
                        bce_loss2 = F.binary_cross_entropy(edge_probs[~nit], HG.masked_edge_labels[~nit])
                        bce_loss += (bce_loss1 + bce_loss2).item()

                        edge_probs = solver(HG, edge_probs, train=False)
                        h_loss1 = hamming_loss(edge_probs[nit], HG.masked_edge_labels[nit])
                        h_loss2 = hamming_loss(edge_probs[~nit], HG.masked_edge_labels[~nit])
                        loss += (h_loss1 + h_loss2).item()


                        TP = torch.sum((HG.masked_edge_labels==1) & (edge_probs>0.5)).item() 
                        TN = torch.sum((HG.masked_edge_labels==0) & (edge_probs<0.5)).item()
                        FP = torch.sum((HG.masked_edge_labels==0) & (edge_probs>0.5)).item() 
                        FN = torch.sum((HG.masked_edge_labels==1) & (edge_probs<0.5)).item()
                        TPs += [TP]; TNs += [TN]; FPs += [FP]; FNs += [FN]
                        num_graphs += 1
                        
                tprs += [sum(TPs)/(sum(TPs)+sum(FNs))]
                fprs += [sum(FPs)/(sum(FPs)+sum(TNs))]
                losses += [loss/num_graphs]
                bce_losses += [bce_loss/num_graphs]
        
        return ts, np.array(losses), np.array(bce_losses), np.array(tprs), np.array(fprs)

    def to_file(self, path: str):
        torch.save(self.anchor_1, path)

    def from_file(self, path: str):
        self.anchor_1 = torch.load(path)
        self.parameter_vectors = [self.anchor_0, self.anchor_1, self.anchor_2]