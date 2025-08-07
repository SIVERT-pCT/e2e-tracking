import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import cm
from src.utils.experiments import ExperimentBase
from src.utils.reporting import experiment_to_table
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

print("Generating performance table ...")

for j, name in enumerate(["pat_lambda_25", "pat_lambda_50", "pat_lambda_75", "ptt"]):
    exp = ExperimentBase.from_file(f"experiments/new/{name}.json")
    experiment_to_table(exp, f"tables/{name}.tex")

colors = cm.YlGnBu(np.linspace(0.3, 0.95, 4))

def create_buffer(n_runs, n_validation_steps):
    buff = np.empty((n_runs, n_validation_steps))
    buff.fill(np.inf)
    return buff

def create_n_buffer(n_buffer, n_runs, n_validation_steps):
    return [create_buffer(n_runs, n_validation_steps) for _ in range(n_buffer)]

def run_to_dfs(event_acc):
    train_pur_df = pd.DataFrame(event_acc.Scalars("Performance/pur_train"))
    train_pur_df = train_pur_df.rename(columns={"value": "pur"})
    train_eff_df = pd.DataFrame(event_acc.Scalars("Performance/eff_train"))
    train_eff_df = train_eff_df.rename(columns={"value": "eff"})
    train_tpr_df = pd.DataFrame(event_acc.Scalars("Performance/tpr_train"))
    train_tpr_df = train_tpr_df.rename(columns={"value": "tpr"})
    train_fpr_df = pd.DataFrame(event_acc.Scalars("Performance/fpr_train"))
    train_fpr_df = train_fpr_df.rename(columns={"value": "fpr"})

    val_pur_df = pd.DataFrame(event_acc.Scalars("Performance/pur_val"))
    val_pur_df = val_pur_df.rename(columns={"value": "pur"})
    val_eff_df = pd.DataFrame(event_acc.Scalars("Performance/eff_val"))
    val_eff_df = val_eff_df.rename(columns={"value": "eff"})
    val_tpr_df = pd.DataFrame(event_acc.Scalars("Performance/tpr_val"))
    val_tpr_df = val_tpr_df.rename(columns={"value": "tpr"})
    val_fpr_df = pd.DataFrame(event_acc.Scalars("Performance/fpr_val"))
    val_fpr_df = val_fpr_df.rename(columns={"value": "fpr"})

    train_df = pd.concat([train_pur_df, train_eff_df, train_tpr_df, train_fpr_df], axis=1)
    train_df = train_df.loc[:,~train_df.columns.duplicated()]

    val_df = pd.concat([val_pur_df, val_eff_df, val_tpr_df, val_fpr_df], axis=1)
    val_df = val_df.loc[:,~val_df.columns.duplicated()]

    return (train_df, val_df)


print("Generating training history figure...")

fig, ax = plt.subplots(2, 4, figsize=(10, 5))

for a in ax.flatten():
    a.set_xlabel("Optimisation step")

for a in ax[:,0]: a.set_ylabel("True pos. rate [%]"); a.set_ylim([99.1, 99.37])
for a in ax[:,1]: a.set_ylabel("False pos. rate [%]"); a.set_ylim([0.16, 0.25]); a.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
for a in ax[:,2]: a.set_ylabel("Track purity [%]"); a.set_ylim([87, 92])
for a in ax[:,3]: a.set_ylabel("Track efficiency [%]"); a.set_ylim([78, 82.5])

for j, name in enumerate(["pat_lambda_25", "pat_lambda_50", "pat_lambda_75", "ptt"]):

    exp = ExperimentBase.from_file(f"experiments/new/{name}.json")

    val_pur, val_eff, val_tpr, val_fpr = create_n_buffer(4, 5, 100)
    train_pur, train_eff, train_tpr, train_fpr = create_n_buffer(4, 5, 100)
    max_indices = np.empty(5)

    for i,  run in enumerate(exp.load_run_logs().runs.values):

        event_acc = EventAccumulator(run)
        event_acc.Reload()

        (train_df, val_df) = run_to_dfs(event_acc)

        max_index = val_df["pur"].values.argmax()
        max_indices[i] = max_index

        val_pur[i, :max_index] = val_df["pur"].values[:max_index] * 100
        val_eff[i, :max_index] = val_df["eff"].values[:max_index] * 100
        val_tpr[i, :max_index] = val_df["tpr"].values[:max_index] * 100
        val_fpr[i, :max_index] = val_df["fpr"].values[:max_index] * 100

        train_pur[i, :max_index] = train_df["pur"].values[:max_index] * 100
        train_eff[i, :max_index] = train_df["eff"].values[:max_index] * 100
        train_tpr[i, :max_index] = train_df["tpr"].values[:max_index] * 100
        train_fpr[i, :max_index] = train_df["fpr"].values[:max_index] * 100

    val_pur = np.ma.masked_array(val_pur, ~np.isfinite(val_pur))
    val_eff = np.ma.masked_array(val_eff, ~np.isfinite(val_eff))
    val_tpr = np.ma.masked_array(val_tpr, ~np.isfinite(val_tpr))
    val_fpr = np.ma.masked_array(val_fpr, ~np.isfinite(val_fpr))

    train_pur = np.ma.masked_array(train_pur, ~np.isfinite(val_pur))
    train_eff = np.ma.masked_array(train_eff, ~np.isfinite(val_eff))
    train_tpr = np.ma.masked_array(train_tpr, ~np.isfinite(val_tpr))
    train_fpr = np.ma.masked_array(train_fpr, ~np.isfinite(val_fpr))

    ax[0, 0].plot(100 * np.arange(train_tpr.shape[1]), train_tpr.mean(axis=0), linewidth=1, color=colors[j])
    ax[0, 0].fill_between(100 * np.arange(train_tpr.shape[1]), train_tpr.min(axis=0), train_tpr.max(axis=0), alpha=0.3, color=colors[j])
    ax[0, 0].vlines(100 * max_indices, 0, 100, linestyle=":", linewidth=1, color=colors[j])

    ax[0, 1].plot(100 * np.arange(train_fpr.shape[1]), train_fpr.mean(axis=0), linewidth=1, color=colors[j])
    ax[0, 1].fill_between(100 * np.arange(train_fpr.shape[1]), train_fpr.min(axis=0), train_fpr.max(axis=0), alpha=0.3, color=colors[j])
    ax[0, 1].vlines(100 * max_indices, 0, 100, linestyle=":", linewidth=1, color=colors[j])

    ax[0, 2].plot(100 * np.arange(train_pur.shape[1]), train_pur.mean(axis=0), linewidth=1, color=colors[j])
    ax[0, 2].fill_between(100 * np.arange(train_pur.shape[1]), train_pur.min(axis=0), train_pur.max(axis=0), alpha=0.3, color=colors[j])
    ax[0, 2].vlines(100 * max_indices, 0, 100, linestyle=":", linewidth=1, color=colors[j])

    ax[0, 3].plot(100 * np.arange(train_eff.shape[1]), train_eff.mean(axis=0), linewidth=1, color=colors[j])
    ax[0, 3].fill_between(100 * np.arange(train_eff.shape[1]), train_eff.min(axis=0), train_eff.max(axis=0), alpha=0.3, color=colors[j])
    ax[0, 3].vlines(100 * max_indices, 0, 100, linestyle=":", linewidth=1, color=colors[j])

    ax[1, 0].plot(100 * np.arange(val_tpr.shape[1]), val_tpr.mean(axis=0), linewidth=1, color=colors[j])
    ax[1, 0].fill_between(100 * np.arange(val_tpr.shape[1]), val_tpr.min(axis=0), val_tpr.max(axis=0), alpha=0.3, color=colors[j])
    ax[1, 0].vlines(100 * max_indices, 0, 100, linestyle=":", linewidth=1, color=colors[j])

    ax[1, 1].plot(100 * np.arange(val_fpr.shape[1]), val_fpr.mean(axis=0), linewidth=1, color=colors[j])
    ax[1, 1].fill_between(100 * np.arange(val_fpr.shape[1]), val_fpr.min(axis=0), val_fpr.max(axis=0), alpha=0.3, color=colors[j])
    ax[1, 1].vlines(100 * max_indices, 0, 100, linestyle=":", linewidth=1, color=colors[j])

    ax[1, 2].plot(100 * np.arange(val_pur.shape[1]), val_pur.mean(axis=0), linewidth=1, color=colors[j])
    ax[1, 2].fill_between(100 * np.arange(val_pur.shape[1]), val_pur.min(axis=0), val_pur.max(axis=0), alpha=0.3, color=colors[j])
    ax[1, 2].vlines(100 * max_indices, 0, 100, linestyle=":", linewidth=1, color=colors[j])

    ax[1, 3].plot(100 * np.arange(val_eff.shape[1]), val_eff.mean(axis=0), linewidth=1, color=colors[j])
    ax[1, 3].fill_between(100 * np.arange(val_eff.shape[1]), val_eff.min(axis=0), val_eff.max(axis=0), alpha=0.3, color=colors[j])
    ax[1, 3].vlines(100 * max_indices, 0, 100, linestyle=":", linewidth=1, color=colors[j])


for a in ax.flatten():
    a.legend([r"PAT ($\lambda = 25$)",
              r"PAT ($\lambda = 50$)",
              r"PAT ($\lambda = 75$)",
              r"PTT"], fontsize="x-small")
    a.grid(True, linestyle=":")

for a, text in zip(ax[:,0], ["Train-set perf.", "Val-set perf."]):
    a.annotate(text, xy=(0, 0.5), xytext=(-a.yaxis.labelpad - 10, 0),
                xycoords=a.yaxis.label, textcoords='offset points',
                ha='right', va='center', rotation=90, weight="bold")


plt.tight_layout()
plt.savefig("figures/training_curve.pdf")