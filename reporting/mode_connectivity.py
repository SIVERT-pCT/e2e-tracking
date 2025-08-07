import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pickle
import numpy as np
import pandas as pd
from typing import List
from os.path import join
from itertools import combinations, product

print("Generating mode connectivity tables...")

names = ["pat_lambda_25", "pat_lambda_50", "pat_lambda_75", "ptt"]
runs = [0, 1, 2, 3, 4]
root_dir = "models/mode"

def load_combination(root_dir, c_tuple):
    try:
        if not isinstance(c_tuple[0][1], int) and \
           not isinstance(c_tuple[1][1], int):
            i, c0 = c_tuple[0]
            j, c1 = c_tuple[1]
        else:
            i = -1; j = -1
            c0 = c_tuple[0]
            c1 = c_tuple[1]
    
        file_path = join(root_dir, f"{c0[0]}_{c0[1]}_{c1[0]}_{c1[1]}.pkl")
        with open(file_path, "rb") as f:
            return (i, j), pickle.load(f)

    except Exception as e:
        if not isinstance(c_tuple[0][1], int) and \
           not isinstance(c_tuple[1][1], int):
            i, c0 = c_tuple[1]
            j, c1 = c_tuple[0]
        else:
            i = -1; j = -1
            c0 = c_tuple[1]
            c1 = c_tuple[0]
            
        file_path = join(root_dir, f"{c0[0]}_{c0[1]}_{c1[0]}_{c1[1]}.pkl")
        with open(file_path, "rb") as f:
            return (i, j), pickle.load(f)


def iterate_files(root_dir: str, names: List[str], runs: List[int]):
    for c_tuple in combinations(enumerate(product(names, runs)), 2):
        yield load_combination(root_dir, c_tuple)


def get_vals(generator, train_loss, eval_loss, fac):
    min_vals = []
    max_vals = []
    mean_vals = []

    for _, f in generator:
        f = f["bezier"] if train_loss != None else f["linear"]
        f = f[train_loss] if train_loss != None else f

        connection = f[f"{eval_loss}_train"]

        if eval_loss in ["tpr", "fpr"]:
            connection *= 100

        min_vals += [connection.min()/fac]
        max_vals += [connection.max()/fac]
        mean_vals += [connection.mean()/fac]

    return min_vals, max_vals, mean_vals

df = pd.DataFrame()
for model in ["pat_lambda_25", "pat_lambda_50", "pat_lambda_75", "ptt"]:
    
    row = dict()
    for train_loss in [None, "hamming", "bce"]:
        #No space for bce , "bce",
        for eval_loss, fac in zip(["hamming", "tpr"], [1e-3, 1, 1e-2]):

            min_vals, max_vals, mean_vals = get_vals(iterate_files(root_dir, [model], runs), 
                                                     train_loss, eval_loss, fac)

            row[f"{eval_loss}_min"] = fr"{np.mean(min_vals):.2f}$\pm${np.std(min_vals):.2f}"
            row[f"{eval_loss}_max"] = fr"{np.mean(max_vals):.2f}$\pm${np.std(max_vals):.2f}"
            row[f"{eval_loss}_mean"] = fr"{np.mean(mean_vals):.2f}$\pm${np.std(mean_vals):.2f}"

        df = pd.concat([df, pd.DataFrame(data=[row])]).reset_index(drop=True)

with open("tables/inter_connectivity.tex", "w") as f:
    df.to_latex(index=False, escape=False, buf=f)