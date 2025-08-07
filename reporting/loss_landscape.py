import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import torch
import numpy as np
from matplotlib import ticker, cm
from matplotlib import pyplot as plt

from src.utils.experiments import ExperimentBase
from src.loss_landscape.slice import generate_loss_lanscape


print("Generating loss landscapes for PAT (hamming) ...")

fig, ax = plt.subplots(3, 4, figsize=(10, 6))

for a in ax.flatten():
    a.set_xlabel(r"$1^{st}$ Eigenvector")
    a.set_ylabel(r"$2^{nd}$ Eigenvector")

for i, name in enumerate(["pat_lambda_25", "pat_lambda_50", "pat_lambda_75"]):
    for run in range(4):
        exp = ExperimentBase.from_file(f"experiments/new/{name}.json")
        v0, v1, XX, YY, losses = generate_loss_lanscape(exp, run, "cuda:1", 50)

        logspace = np.logspace(np.log10(losses.min()), np.log10(losses.max()), 25)
        cs_f = ax[i, run].contourf(XX, YY, losses, logspace, cmap="GnBu") #, 20
        cs = ax[i, run].contour(XX, YY, losses, logspace, colors='black', linewidths=0.2)
        fmt = ticker.LogFormatterMathtext()
        fmt.create_dummy_axis()
        ax[i, run].clabel(cs, cs.levels[::2], inline=False, fmt=lambda x: "{:.2f}".format(x*100), fontsize=6)
        ax[i, run].scatter(0, 0, color="red", s=25, marker="*")


for a, text in zip(ax[:,0].flatten(), [r"PAT $(\lambda = 25)$",
                                       r"PAT $(\lambda = 50)$",
                                       r"PAT $(\lambda = 75)$"]):
    a.annotate(text, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 10, 0),
            xycoords=a.yaxis.label, textcoords='offset points',
            ha='right', va='center', rotation=90, weight="bold")

plt.tight_layout(pad=0.7)
plt.savefig("figures/landscapes_pat.png", dpi=300)
plt.savefig("figures/landscapes_pat.pdf")

print("Generating loss landscapes for PTT (hamming & bce) ...")

fig, ax = plt.subplots(2, 4, figsize=(10, 4.5))

for a in ax.flatten():
    a.set_xlabel(r"$1^{st}$ Eigenvector")
    a.set_ylabel(r"$2^{nd}$ Eigenvector")

for i in range(4):
    for j, loss_type in enumerate(["hamming", "bce"]):
        exp = ExperimentBase.from_file(f"experiments/new/ptt.json")
        v0, v1, XX, YY, losses = generate_loss_lanscape(exp, i, "cuda:1", 50, loss_type=loss_type)

        logspace = np.logspace(np.log10(losses.min()), np.log10(losses.max()), 25)
        cs_f = ax[j, i].contourf(XX, YY, losses, logspace, cmap="GnBu") #or YlGnBu
        cs = ax[j, i].contour(XX, YY, losses, logspace, colors='black', linewidths=0.2)
        fmt = ticker.LogFormatterMathtext()
        fmt.create_dummy_axis()
        ax[j, i].clabel(cs, cs.levels[::4], inline=False, fmt=lambda x: "{:.2f}".format(x*100), fontsize=7)
        ax[j, i].scatter(0, 0, color="red", s=25, marker="*")


for a, text in zip(ax[:,0].flatten(), ["PTT (Hamming)", "PTT (BCE)"]):
    a.annotate(text, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 10, 0),
                xycoords=a.yaxis.label, textcoords='offset points',
                ha='right', va='center', rotation=90, weight="bold")

plt.tight_layout()
plt.savefig("figures/landscapes_ptt.pdf")   