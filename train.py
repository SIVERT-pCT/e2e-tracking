from src.utils.experiments import ExperimentBase, CheckpointConfig
import torch
import numpy as np
from matplotlib import pyplot as plt
from src.utils.reporting import experiment_to_table


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, help='.json file as string containing the experiment definition', required=True)
parser.add_argument('-d', '--device', type=str, help='Cuda device used for training (cuda:#)', required=True)
args = parser.parse_args()

exp = ExperimentBase.from_file(args.experiment)
exp.dataset.generate_from_config(args.device)
exp.train_model(args.device)
exp.evaluate_model(args.device)