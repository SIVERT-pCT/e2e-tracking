import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from src.utils.experiments import ExperimentBase
import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from src.supervised.combinatorial.lsa import CombinatorialSolver
from matplotlib import pyplot as plt
from matplotlib import cm
import itertools

def initialize_mode(exp_file, phantom, events):
    exp = ExperimentBase.from_file(exp_file)

    run = 0; device="cuda"; early_stopping=True
    

    network = exp.model.load_from_config(run, device, early_stopping)
    dataset = exp.dataset.load_test_from_config(phantom, events, device)
    solver = None if exp.solver == None else exp.solver.generate_from_config()
    transforms = exp.transforms.generate_from_config()
    if solver ==None: solver = CombinatorialSolver(1.0, 0.0, False)
    return network, dataset, solver, transforms


network, dataset, solver, transforms = initialize_mode("experiments/hit_graph/ptt.json", "water_100", 100)

def flatten(it):
    return list(itertools.chain.from_iterable(it))

def eval_affected_hg(G):
    HG_, LG_ = transforms(G)
    network(HG_)

def eval_all_hg(G):
    HG_, LG_ = transforms(G)
    edge_logits = network(HG_)
    edge_probs = torch.sigmoid(edge_logits)
    edge_probs = solver(HG_, edge_probs, train=True)
    mask = edge_probs > 0.5

    edge_index_masked = HG_.masked_edge_index[:,mask]
    HG_.edge_index = edge_index_masked
    HG_.edge_index = HG_.edge_index[[1, 0]]

    nx_graph = to_networkx(HG_)

    roots  = [v for v, d in nx_graph.in_degree() if d == 0]
    leaves = [v for v, d in nx_graph.out_degree() if d == 0]
    tracks = []

    for root in roots:
        paths = nx.all_simple_paths(nx_graph, root, leaves)
        tracks.extend(paths)

def eval_affected_lg(G):
    HG_, LG_ = transforms(G)
    network(LG_)

def eval_all_lg(G):
    HG_, LG_ = transforms(G)
    edge_logits = network(LG_)
    edge_probs = torch.sigmoid(edge_logits)
    edge_probs = solver(HG_, edge_probs, train=True)
    mask = edge_probs > 0.5

    edge_index_masked = HG_.masked_edge_index[:,mask]
    HG_.edge_index = edge_index_masked
    HG_.edge_index = HG_.edge_index[[1, 0]]

    nx_graph = to_networkx(HG_)

    roots  = [v for v, d in nx_graph.in_degree() if d == 0]
    leaves = [v for v, d in nx_graph.out_degree() if d == 0]
    tracks = []

    for root in roots:
        paths = nx.all_simple_paths(nx_graph, root, leaves)
        tracks.extend(paths)

def benchmark(fun, graph):
    #Warmup runs
    for i in range(3):
        fun(graph)

    measurements = []

    for i in range(10):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fun(graph)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        measurements += [start.elapsed_time(end)]
        
    return measurements

phantom = "water_100"
events=150
exp_file = "experiments/hit_graph/ptt.json"

colors = cm.YlGnBu(np.linspace(0.3, 0.95, 2))
fig, ax = plt.subplots(1, 3, figsize=(10, 2.5), sharey=True)

ax[0].set_title("100 mm Water Phantom")
ax[1].set_title("150 mm Water Phantom")
ax[2].set_title("200 mm Water Phantom")


for i, phantom in enumerate(["water_100", "water_150", "water_200"]): #, "water_150", "water_200"
    all_measurements_a = []
    all_measurements_b = []
    for events in [50, 100, 150]:
        network, dataset, solver, transforms = initialize_mode("experiments/hit_graph/ptt.json", phantom, events)
        
        measurements_a = []
        measurements_b = []
        for f, HG in enumerate(dataset):
            measurements_a += [benchmark(eval_affected_hg, HG)]
            measurements_b += [benchmark(eval_all_hg, HG)]

        all_measurements_a += [flatten(measurements_a)]
        all_measurements_b += [flatten(measurements_b)]
    
    ax[i].plot([50, 100, 150],[np.mean(m) for m in all_measurements_a], marker="o", label=r"$\mathcal{G}_H$ - Pred.", color=colors[0])
    ax[i].plot([50, 100, 150],[np.mean(b) for b in all_measurements_b], marker="o", linestyle="--", label=r"$\mathcal{G}_H$ - Reco.", color=colors[0])
    ax[i].set_xticks([50, 100, 150])


for i, phantom in enumerate(["water_100", "water_150", "water_200"]): #, "water_150", "water_200"
    all_measurements_a = []
    all_measurements_b = []
    for events in [50, 100, 150]:
        network, dataset, solver, transforms = initialize_mode("experiments/line_graph/ptt.json", phantom, events)

        measurements_a = []
        measurements_b = []

        for f, HG in enumerate(dataset):
            measurements_a += [benchmark(eval_affected_lg, HG)]
            measurements_b += [benchmark(eval_all_lg, HG)]

        all_measurements_a += [flatten(measurements_a)]
        all_measurements_b += [flatten(measurements_b)]

    ax[i].plot([50, 100, 150], [np.mean(m) for m in all_measurements_a], marker="o", label=r"$\mathcal{G}_L$ - Pred.", color=colors[1])
    ax[i].plot([50, 100, 150], [np.mean(b) for b in all_measurements_b], linestyle="--", marker="o", label=r"$\mathcal{G}_L$ - Reco.", color=colors[1])
    ax[i].set_xticks([50, 100, 150])

for a in ax: a.legend(fontsize="small")
ax[0].set_ylabel("Avg. runtime [ms]")
for a in ax: a.set_xlabel("Particle density [$p^+/F$]")
plt.tight_layout()
plt.savefig("figures/runtimes_hg_lg.pdf")


