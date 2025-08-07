import os
import subprocess
import argparse
from itertools import combinations, product


parser = argparse.ArgumentParser()
parser.add_argument("--type", "-t")

args = parser.parse_args()

models = ["ptt", "pat_lambda_25", "pat_lambda_50", "pat_lambda_75"]
runs = [0, 1, 2, 3, 4]

curr_dir = os.getcwd()

for combination in combinations(product(models, runs), 2):
    command = [
        "sbatch", f"{curr_dir}/run_single_repfun.slurm",
        "--exp_a", combination[0][0],
        "--run_a", str(combination[0][1]),
        "--exp_b", combination[1][0],
        "--run_b", str(combination[1][1]),
        "--cwd", curr_dir, 
        "--type", args.type]

    subprocess.run(command)
