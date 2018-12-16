#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=classifier_eval_all
#SBATCH --mail-type=END
#SBATCH --mail-user=cer446@nyu.edu
#SBATCH --output=classifier_eval_all_lm_4_bs32.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module purge
module load python3/intel/3.6.3
source ~/pyenv/fastai.0.7.0/bin/activate

python eval_clas.py classifier/Obscenity 0 --lm-id 4 --clas-id obscenity --bs 32
python eval_clas.py classifier/hatespeech 0 --lm-id 4 --clas-id hatespeech --bs 32
python eval_clas.py classifier/namecalling 0 --lm-id 4 --clas-id namecalling --bs 32
python eval_clas.py classifier/negprejudice 0 --lm-id 4 --clas-id negprejudice --bs 32
python eval_clas.py classifier/noneng 0 --lm-id 4 --clas-id noneng --bs 32
python eval_clas.py classifier/porn 0 --lm-id 4 --clas-id porn --bs 32
python eval_clas.py classifier/stereotypes 0 --lm-id 4 --clas-id stereotypes --bs 32
python eval_clas.py classifier/Threat 0 --lm-id 4 --clas-id Threat --bs 32

