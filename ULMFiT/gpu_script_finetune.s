#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=18:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=language_model_6
#SBATCH --mail-type=END
#SBATCH --mail-user=cer446@nyu.edu
#SBATCH --output=lm_output_v4.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module purge
module load python3/intel/3.6.3
source ~/pyenv/fastai.0.7.0/bin/activate

python finetune_lm.py complete_language_model wt103 --cl 30 --lm-id 4 --dropmult 0.70 --use-discriminative True --early-stopping True 
