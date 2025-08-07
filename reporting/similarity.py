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
from matplotlib import pyplot as plt
from itertools import combinations, product
from mpl_toolkits.axes_grid1 import make_axes_locatable

print("Generating similarity plots...")

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

def plot_similarity_matrix(ax, x):
    x_mask = np.zeros_like(x).astype(np.bool_)
    x_mask[np.triu_indices(n=x.shape[0])] = True
    x_masked = np.ma.array(x, mask=x_mask)

    ax.imshow(x_masked)
    im = ax.imshow(x, cmap="GnBu")
    ax.tick_params('x', length=15, width=1)
    ax.set_xticks([4.5, 9.5, 14.5, 19.5])

    ax.set_xticks([2.2, 7.2, 12.2, 17.25], minor=True)
    ax.set_xticklabels(['$\lambda = 25$', '$\lambda = 50$', '$\lambda = 75$', 'PTT'], minor=True)
    ax.set_xticklabels('')
    ax.tick_params(axis='x', which='minor', pad=2)

    ax.tick_params('y', length=15, width=1)
    ax.set_yticks([4.5, 9.5, 14.5, 19.5])
    ax.set_yticklabels('')

    ax.set_yticks([0.2, 5.5, 10.5, 16.4], minor=True)
    ax.set_yticklabels(['$\lambda = 25$', '$\lambda = 50$', '$\lambda = 75$', 'PTT'], minor=True)
    ax.tick_params(axis='y', which='minor', pad=2, labelrotation=90)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax, extend="both")

def plot_boxes(x, ax):
    short_names = [r"$\lambda=25$", r"$\lambda=50$", r"$\lambda=75$", "ptt"]

    groups = {}
    for i in range(4):
        for j in range(i+1):
            block = x[i*5:i*5+5, j*5:j*5+5].flatten()
            elements = block[~np.isinf(block)]
            groups[f"{short_names[i]}" + r"$\rightarrow$" + f"{short_names[j]}"] = elements

    bplot = ax.boxplot(groups.values(), patch_artist=True)
    x = np.stack(list(map(np.median, groups.values())))
    norm = mpl.colors.Normalize(vmin=x.min(), vmax=x.max())

    for patch, avg in zip(bplot["boxes"], groups.values()):
        patch.set_facecolor(plt.cm.GnBu(norm(np.median(avg))))

    for patch in bplot["medians"]:
        patch.set_color("black")

    ax.set_xticklabels(groups.keys(), rotation=90)


fig, ax = plt.subplots(2, 4, figsize=(12, 6.5), gridspec_kw={'height_ratios': [2.5, 1]})
for a, x in zip(ax[0], [CKA_R1, CKA_O, CKA_R2, M]):
    plot_similarity_matrix(a, x)

for a, name in zip(ax[0,:3].flatten(), ["R1", "O", "R2"]):
    a.set_title(r"$\overline{CKA}$: " + fr"{name}-module $(\uparrow)$")

ax[0,-1].set_title(r"$m_{\min\max}$ $(\downarrow)$")

for a, x in zip(ax[1], [CKA_R1, CKA_O, CKA_R2, M]):
    plot_boxes(x, a)
    a.grid(True, linestyle=":")

plt.tight_layout()
plt.savefig("figures/model_run_similarities.pdf")


