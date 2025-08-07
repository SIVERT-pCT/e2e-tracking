import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


import pickle
import numpy as np
from os.path import join
import matplotlib as mpl
import pandas as pd
from scipy.stats import ttest_ind
from matplotlib import ticker
from matplotlib import pyplot as plt
from itertools import combinations, product
from mpl_toolkits.axes_grid1 import make_axes_locatable

print("Generating significance plots...")

names = ["pat_lambda_25", "pat_lambda_50", "pat_lambda_75", "ptt"]
runs = [0, 1, 2, 3, 4]
root_dir = "models/cka"

M = np.zeros((20, 20))
CKA_R1 = np.zeros((20, 20))
CKA_O = np.zeros((20, 20))
CKA_R2 = np.zeros((20, 20))

np.fill_diagonal(CKA_R1, np.inf)
np.fill_diagonal(CKA_O, np.inf)
np.fill_diagonal(CKA_R2, np.inf)
np.fill_diagonal(M, np.inf)


for c_tuple in combinations(enumerate(product(names, runs)), 2):
    i, c0 = c_tuple[0]
    j, c1 = c_tuple[1]
    file_path = join(root_dir, f"{c0[0]}_{c0[1]}_{c1[0]}_{c1[1]}.pkl")
    with open(file_path, "rb") as f:
        res = pickle.load(f)
        M[i,j] = res["m_minmax"]
        CKA_R1[i,j] = res["R1"].mean()
        CKA_O[i,j] = res["O"].mean()
        CKA_R2[i,j] = res["R2"].mean()

        M[j,i] = M[i,j]
        CKA_R1[j,i] = CKA_R1[i,j]
        CKA_O[j,i] = CKA_O[i,j]
        CKA_R2[j,i] = CKA_R2[i,j]

groups = []
names = []
short_names = [r"$\lambda=25$", r"$\lambda=50$", r"$\lambda=75$", "ptt"]

for i in range(4):
    for j in range(i+1):
        block = M[i*5:i*5+5, j*5:j*5+5].flatten()
        elements = block[~np.isinf(block)]
        groups += [elements] 
        names += [f"{short_names[i]}" + r"$\rightarrow$" + f"{short_names[j]}"]



statistic = np.zeros((10, 10))
pvalue = np.zeros((10, 10))

for i in range(10):
    for j in range(10):
        a = groups[i]
        b = groups[j]
        res = ttest_ind(a, b, permutations=1000, equal_var=False, alternative="less")
        statistic[i, j] = res.statistic
        pvalue[i, j] = res.pvalue

[len(g) for g in groups]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

cm = ax[0].imshow(statistic, cmap="GnBu")
plt.colorbar(cm, ax=ax[0])

cm = ax[1].imshow(pvalue, cmap="GnBu")
plt.colorbar(cm, ax=ax[1])


ax[0].set_yticks(np.arange(10))
ax[0].set_yticklabels(names)

ax[0].set_xticks(np.arange(10))
ax[0].set_xticklabels(names)

ax[1].set_xticks(np.arange(10))
ax[1].set_xticklabels(names)

ax[0].tick_params(axis='x', which='major', pad=2, labelrotation=90)
ax[1].tick_params(axis='x', which='major', pad=2, labelrotation=90)

ax[0].set_title("t-statistic (a < b)")
ax[1].set_title("p-value (a < b)")


plt.tight_layout()
plt.savefig("figures/statistic_similarity.pdf")


