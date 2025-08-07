import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from src.utils.experiments import ExperimentBase
import torch
from torch import nn
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from src.supervised.combinatorial.lsa import CombinatorialSolver
from matplotlib import pyplot as plt
from matplotlib import cm
from fvcore.nn import FlopCountAnalysis
from src.supervised.gnn.models import HGInteractionNetwork

def initialize(exp_file, phantom, events):
    exp = ExperimentBase.from_file(exp_file)

    run = 0; device="cuda"; early_stopping=True
    
    network = exp.model.load_from_config(run, device, early_stopping)
    dataset = exp.dataset.load_test_from_config(phantom, events, device)
    solver = None if exp.solver == None else exp.solver.generate_from_config()
    transforms = exp.transforms.generate_from_config()
    if solver ==None: solver = CombinatorialSolver(1.0, 0.0, False)
    return network, dataset, solver, transforms


class SmallHGWrapper:
    x: torch.Tensor
    masked_edge_index: torch.Tensor
    masked_edge_attr: torch.Tensor

    def __init__(self, x, masked_edge_index, masked_edge_attr):
        self.x = x
        self.masked_edge_index = masked_edge_index
        self.masked_edge_attr = masked_edge_attr

class SmallLGWrapper:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor

    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

class HGInteractionNetworkWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, masked_edge_index, masked_edge_attr):
        G = SmallHGWrapper(x, masked_edge_index, masked_edge_attr)
        return self.network(G)
    
class LGInteractionNetworkWrapper(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x, edge_index, edge_attr):
        G = SmallLGWrapper(x, edge_index, edge_attr)
        return self.network(G)


results = dict()

for exp in ["experiments/hit_graph/ptt.json", "experiments/line_graph/ptt.json"]:
    results[exp] = np.empty((3, 3))
    for i, particles in enumerate([50, 100, 150]):
        for j, wpt in enumerate([100, 150, 200]):
            network, dataset, solver, transforms = initialize(exp, f"water_{wpt}", particles)
            flops_list = []
            for G in dataset:

                HG, LG = transforms(G)

                if isinstance(network, HGInteractionNetwork):
                    flops = FlopCountAnalysis(HGInteractionNetworkWrapper(network), (HG.x, HG.masked_edge_index, HG.masked_edge_attr))
                    flops_list += [flops.total()/1e9]
                else:
                    flops = FlopCountAnalysis(LGInteractionNetworkWrapper(network), (LG.x, LG.edge_index, LG.edge_attr))
                    flops_list += [flops.total()/1e9]
                    
            results[exp][i,j] = np.array(flops_list).mean()


colors = cm.YlGnBu(np.linspace(0.3, 0.95, 2))
fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(10, 2.5))
for i in range(3):
    ax[i].plot([50, 100, 150], results["experiments/line_graph/ptt.json"][:,i], marker="o", color=colors[1])
    ax[i].plot([50, 100, 150], results["experiments/hit_graph/ptt.json"][:,i], linestyle="--", marker="o", color=colors[0])
    ax[i].legend([r"Pred. ($\mathcal{G}_H$)", 
                  r"Pred. ($\mathcal{G}_L$)"],fontsize="small")
    
    ax[i].set_xticks([50, 100, 150])

ax[0].set_ylabel("GFLOP")
for a in ax: a.set_xlabel("Particle density [p$^+$/F]")

ax[0].set_title("100 mm Water Phantom")
ax[1].set_title("150 mm Water Phantom")
ax[2].set_title("200 mm Water Phantom")

plt.tight_layout()
plt.savefig("figures/flops_hg_lg.pdf")


