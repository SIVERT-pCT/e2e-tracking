import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from src.utils.experiments import ExperimentBase
from math import floor
from torch_geometric.utils import to_networkx
from src.supervised.combinatorial.lsa import CombinatorialSolver
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np

exp = ExperimentBase.from_file("experiments/line_graph/ptt.json")
transforms = exp.transforms.generate_from_config()

nodes_hg_wpt = []
edges_hg_wpt = []
nodes_lg_wpt = []
edges_lg_wpt = []

for particles in [50, 100, 150]:

    nodes_hg_particles = []
    edges_hg_particles = []
    nodes_lg_particles = []
    edges_lg_particles = []

    for wpt in [100, 150, 200]: 
        
        dataset = exp.dataset.load_test_from_config(f"water_{wpt}", particles)
        nodes_hg = []
        edges_hg = []
        nodes_lg = []
        edges_lg = []

        for G in dataset:
            HG, LG = transforms(G)
            nodes_hg += [HG.num_nodes]
            edges_hg += [HG.masked_edge_index.shape[1]]
            nodes_lg += [LG.num_nodes]
            edges_lg += [LG.num_edges]

        nodes_hg_particles += [f"{floor(np.array(nodes_hg).mean())}$\pm${floor(np.array(nodes_hg).std())}"]
        edges_hg_particles += [f"{floor(np.array(edges_hg).mean())}$\pm${floor(np.array(edges_hg).std())}"]
        nodes_lg_particles += [f"{floor(np.array(nodes_lg).mean())}$\pm${floor(np.array(nodes_lg).std())}"]
        edges_lg_particles += [f"{floor(np.array(edges_lg).mean())}$\pm${floor(np.array(edges_lg).std())}"]

    nodes_hg_wpt += [nodes_hg_particles]
    edges_hg_wpt += [edges_hg_particles]
    nodes_lg_wpt += [nodes_lg_particles]
    edges_lg_wpt += [edges_lg_particles]



#Write hit graph results to file
df_nodes_hg = pd.DataFrame(nodes_hg_wpt, columns=["n_100", "n_150", "n_200"])
df_edges_hg = pd.DataFrame(edges_hg_wpt, columns=["e_100", "e_150", "e_200"])

df = pd.concat((df_nodes_hg, df_edges_hg), axis=1)
df = df.reindex(columns=["n_100", "e_100", "n_150", "e_150", "n_200", "e_200"])

with open("tables/graph_sizes_hg.tex", "w") as f:
    df.to_latex(index=False, escape=False, buf=f)


#Write line graph results to file
df_nodes_lg = pd.DataFrame(nodes_lg_wpt, columns=["n_100", "n_150", "n_200"])
df_edges_lg = pd.DataFrame(edges_lg_wpt, columns=["e_100", "e_150", "e_200"])

df = pd.concat((df_nodes_lg, df_edges_lg), axis=1)
df = df.reindex(columns=["n_100", "e_100", "n_150", "e_150", "n_200", "e_200"])

with open("tables/graph_sizes_lg.tex", "w") as f:
    df.to_latex(index=False, escape=False, buf=f)


