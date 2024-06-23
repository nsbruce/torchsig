#!/bin/bash

#SBATCH --job-name=sig53
#SBATCH --time=03-00:00:00
#SBATCH --account=def-msteve
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100l:4              # Number of GPUs (per node)
#SBATCH --mem=0               # memory (per node)
#SBATCH --mail-user=nsbruce@uvic.ca #bmoa@uvic.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023
module load python/3.10
source /project/def-msteve/nsbruce/RFI/torchsig/venv/bin/activate
rstrt=${1:-1}
ckpt_cmd=""
if [ $rstrt -eq 1 ] # if restart is set, use the latest model to continue the training
then
   ckpt_cmd="--ckpt_file=`ls -Art ./saved_models/*.ckpt| tail -n 1`" 
fi

export NCCL_BLOCKING_WAIT=1  # Aovid NCCL timeout errors
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
# source /opt/software/pysig/bin/activate
echo srun python ./train.py --batch_size=64 --val_every=1 --epochs=100 --num_workers=$((SLURM_CPUS_PER_TASK)) --device='cuda' ${ckpt_cmd}
srun python ./train.py --batch_size=64 --val_every=1 --epochs=100 --num_workers=$((SLURM_CPUS_PER_TASK)) --device='cuda' ${ckpt_cmd}
