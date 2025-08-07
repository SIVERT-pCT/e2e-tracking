from src.supervised.combinatorial.lsa import CombinatorialSolver
from src.utils.dataset import HitGraphDataset
from src.utils.experiments import ExperimentBase
from src.supervised.gnn.models import LGInteractionNetwork, HGInteractionNetwork
from typing import Dict, Tuple, Callable, List
from torch.utils.hooks import RemovableHandle
from torch import nn
import numpy as np
import torch
from tqdm import tqdm
from src.utils.experiments import set_random_seeds
from src.utils.eval import subset

from src.utils.transforms.graph import CustomGraphTransform
from src.loss_landscape.similarity.utils import generate_gram_matrix, iterate_minibatches, hisc, is_relevant_layer

def assert_same_preprocessing(exp_a, exp_b):
    return True

def assert_same_data(exp_a, exp_b):
    return True

class RepFunSimilarity:
    def __init__(self, network_a: LGInteractionNetwork, network_b:LGInteractionNetwork, 
                 dataset: HitGraphDataset, transforms: CustomGraphTransform, 
                 solver: CombinatorialSolver):
        
        self.network_a = network_a
        self.network_b = network_b
        self.dataset = dataset
        self.transforms = transforms
        self.solver = solver
        self.modules = ["R1", "O", "R2"]

        self.N = 100
        self.indices = np.arange(len(dataset))
        self.indices = self.indices[::int(np.ceil(len(self.indices) / self.N ))]

    @staticmethod
    def from_custom(exp_a: ExperimentBase, exp_b: ExperimentBase, run_a: int, run_b: int, device: str, on_test_set: bool = False):
        network_a = exp_a.model.load_from_config(run_a, device)
        network_b = exp_b.model.load_from_config(run_b, device)

        assert_same_preprocessing(exp_a, exp_b)
        assert_same_data(exp_a, exp_b)

        transforms = exp_a.transforms.generate_from_config() 
        dataset = exp_a.dataset.load_train_from_config(device) \
                        if not on_test_set \
                        else exp_a.dataset.load_test_from_config("water_100", 150, device)
        
        solver = CombinatorialSolver(1, 1, False)
        return RepFunSimilarity(network_a, network_b, dataset, transforms, solver)

    @staticmethod
    def from_experiments(exp_a: ExperimentBase, exp_b: ExperimentBase, run: int, device: str, on_test_set: bool = False):
        network_a = exp_a.model.load_from_config(run, device)
        network_b = exp_b.model.load_from_config(run, device)

        assert_same_preprocessing(exp_a, exp_b)
        assert_same_data(exp_a, exp_b)

        transforms = exp_a.transforms.generate_from_config()
        dataset = exp_a.dataset.load_train_from_config(device) \
                        if not on_test_set \
                        else exp_a.dataset.load_test_from_config("water_100", 150, device)
        solver = CombinatorialSolver(1, 1, False)
        return RepFunSimilarity(network_a, network_b, dataset, transforms, solver)

    @staticmethod
    def from_runs(exp: ExperimentBase, run_a: int, run_b: int, device: str, on_test_set: bool = False):
        network_a = exp.model.load_from_config(run_a, device)
        network_b = exp.model.load_from_config(run_b, device)

        transforms = exp.transforms.generate_from_config()
        dataset = exp.dataset.load_train_from_config(device) \
                        if not on_test_set \
                        else exp.dataset.load_test_from_config("water_100", 150, device)
        solver = CombinatorialSolver(1, 1, False)
        return RepFunSimilarity(network_a, network_b, dataset, transforms, solver)
    
    def __log_layer_hook(self, model_name: str, module_name: str, layer_name: str) -> Callable:
        def hook(layer: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self.features[model_name][module_name][layer_name] = output
        return hook

    def get_cka_similarities(self) -> Dict[str, np.ndarray]:
        hsic_matrices = {}
        num_batches = {"pred": 0, "R1": 0, "O": 0, "R2": 0}
        self.features = {"A": dict([(name, dict()) for name in self.modules]), 
                         "B": dict([(name, dict()) for name in self.modules])}
        
        with torch.inference_mode():
            set_random_seeds(0)
            self.network_a.eval()
            self.network_b.eval()

            hooks: List[RemovableHandle] = []

            ##Init procedure
            for module in self.modules:
                module_a: nn.Module = getattr(self.network_a, module)
                module_b: nn.Module = getattr(self.network_b, module)

                i = 0; j = 0
                for name, layer in module_a.named_modules():
                    if is_relevant_layer(layer):
                        hooks.append(layer.register_forward_hook(self.__log_layer_hook("A", module, name)))
                        i += 1
                for name, layer in module_b.named_modules():
                    if is_relevant_layer(layer):
                        hooks.append(layer.register_forward_hook(self.__log_layer_hook("B", module, name)))
                        j += 1

                hsic_matrices[module] = torch.zeros((i, j, 3), device=self.dataset.device)
            
            
            for G in tqdm(subset(self.dataset, self.indices), total=self.N):
                HG, LG = self.transforms(G)
                _ = self.network_a(LG)    
                _ = self.network_b(LG)  
            
                for module in self.modules:
                    for _ in range(5):
                        for i, feat_a in enumerate(self.features["A"][module].values()):
                            for j, feat_b in enumerate(self.features["B"][module].values()):
                                feat_a = feat_a[:1024*80]
                                feat_b = feat_b[:1024*80]
                                for X, Y in iterate_minibatches(feat_a, feat_b, 1024, shuffle=True):
                                    K = generate_gram_matrix(X)
                                    L = generate_gram_matrix(Y)
                                    
                                    hsic_matrices[module][i,j,0] += hisc(K,K)
                                    hsic_matrices[module][i,j,1] += hisc(K,L)
                                    hsic_matrices[module][i,j,2] += hisc(L,L)   
                                    num_batches[module] += 1                       

            for module in self.modules:
                hsic_matrices[module] = hsic_matrices[module]/num_batches[module]
                hsic_matrices[module] = hsic_matrices[module][:,:,1] / (hsic_matrices[module][:,:,0].sqrt() * hsic_matrices[module][:,:,2].sqrt())
                hsic_matrices[module] = hsic_matrices[module].cpu().numpy()
                assert not np.any(np.isnan(hsic_matrices[module])), "HSIC computation resulted in NANs"
                
            for h in hooks: h.remove() 
            return hsic_matrices

    def get_disagreement(self) -> Tuple[float, float]:
        q_err_a, q_err_b, m, N = [0] * 4

        for G in tqdm(self.dataset):
            HG, LG = self.transforms(G)
            pred_a = self.solver(HG, self.network_a(LG))    
            pred_b = self.solver(HG, self.network_b(LG))  
            q_err_a += (pred_a != HG.masked_edge_labels).sum()
            q_err_b += (pred_b != HG.masked_edge_labels).sum()
            m += (pred_a != pred_b).sum()
            N += pred_a.shape[0]

        q_err_a: torch.Tensor = q_err_a/N
        q_err_b: torch.Tensor = q_err_b/N
        m: torch.Tensor = m/N

        m_min = torch.abs(q_err_a - q_err_b)
        m_max = torch.minimum((q_err_a + q_err_b), torch.tensor([1], device=m_min.device))
        m_minmax = (m - m_min)/(m_max - m_min) 
        return m.item(), m_minmax.item()