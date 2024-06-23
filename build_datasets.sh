#!/bin/bash
#SBATCH --account=def-msteve
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=8-0:0:0
#SBATCH --job-name=impaired_train

module load StdEnv/2023
module load python/3.10

source /project/def-msteve/nsbruce/RFI/torchsig/venv/bin/activate
python3 /project/def-msteve/nsbruce/RFI/torchsig/scripts/generate_sig53.py --all=False --root=/project/def-msteve/torchsig/sig53 --config-index=2
# python3 /project/def-msteve/nsbruce/RFI/torchsig/scripts/generate_wideband_sig53.py --all=True --root=/project/def-msteve/torchsig/sig53_wideband
