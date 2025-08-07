#!/bin/bash
#SBATCH --gres=gpu:1 
#SBATCH --mem=8G 
#SBATCH --time=12:00:00

module load anaconda3
source $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate gnn

python "${10}/run_single_repfun.py" $@
