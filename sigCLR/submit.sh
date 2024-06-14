#!/bin/bash

#SBATCH --job-name=sig53
#SBATCH --time=03-00:00:00
#SBATCH --account=def-msteve
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:v100l:4              # Number of GPUs (per node)
#SBATCH --mem=0               # memory (per node)
#SBATCH --mail-user=nsbruce@uvic.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023
module load python/3.10
source /project/def-msteve/nsbruce/RFI/torchsig/venv/bin/activate

# source /opt/software/pysig/bin/activate
python ./train.py --batch_size=64 --val_every=10 --epochs=50000
