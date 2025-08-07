import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import torch
import numpy as np
import matplotlib
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.utils.experiments import ExperimentBase
from src.utils.transforms.graph import ToMaskedTGraph
from src.utils.eval import subset

exp = ExperimentBase.from_file("experiments/new/pat_lambda_25.json")
transforms = exp.transforms.generate_from_config()
dataset = exp.dataset.load_train_from_config("cuda:1")

N = 100
indices = np.arange(len(dataset))
indices = indices[::int(np.ceil( len(dataset) / N ))]

all_f_tracker = []
all_f_calorim = []
all_e_tracker = []
all_e_calorim = []

thetas = torch.arange(0, 1.6, 0.05)
for theta in tqdm(thetas):
    transform = ToMaskedTGraph(theta, theta)
    f_calorim = 0
    f_tracker = 0
    e_tracker = 0
    e_calorim = 0
    tf_calorim = 0
    tf_tracker = 0

    for G in subset(dataset, indices):
        HG = transform(G)
        hg_tmask = HG.next_is_tracker[HG.masked_edge_index[1]]
        g_tmask = HG.next_is_tracker[HG.edge_index[1]]

        f_tracker += (HG.edge_labels[g_tmask].sum() - HG.masked_edge_labels[hg_tmask].sum())/HG.edge_labels[g_tmask].sum() * 100
        f_calorim += (HG.edge_labels[~g_tmask].sum() - HG.masked_edge_labels[~hg_tmask].sum())/HG.edge_labels[~g_tmask].sum() * 100

        e_tracker += (HG.edge_labels[g_tmask].numel() - HG.masked_edge_labels[hg_tmask].numel())/HG.edge_labels[g_tmask].numel() * 100
        e_calorim += (HG.edge_labels[~g_tmask].numel() - HG.masked_edge_labels[~hg_tmask].numel())/HG.edge_labels[~g_tmask].numel() * 100

    all_f_tracker += [f_tracker.item()/N]
    all_f_calorim += [f_calorim.item()/N]
    all_e_tracker += [e_tracker/N]
    all_e_calorim += [e_calorim/N]


cmap = matplotlib.cm.get_cmap('YlGnBu')
c0, c1 = cmap(0.4), cmap(0.7)

plt.plot(thetas * 1000, all_f_calorim, marker="o", markersize=4, color="black")
plt.plot(thetas * 1000, all_e_calorim, marker="o", markersize=4, linestyle=":", color="black")

plt.plot(thetas * 1000, all_f_tracker, marker="d", markersize=4, color=c1)
plt.plot(thetas * 1000, all_e_tracker, marker="d", markersize=4, linestyle=":", color=c1)

plt.ylabel("Fraction [%]")
plt.xlabel("Edge threshold [mrad]")

plt.yticks(torch.arange(0, 100+1, 10.0))
plt.xticks(torch.arange(0, 1605, 100))
plt.gcf().set_size_inches(10, 4)

plt.legend([r"True edges rem. ($\theta_{d}$)",
            r"Total edges rem. ($\theta_{d}$)",
            r"True edges rem. ($\theta_{t})$",
            r"Total edges rem. ($\theta_{t}$)"])

plt.grid(True, linestyle=":")
plt.savefig("figures/edge_filter_hit_graph.pdf")


