#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=16:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=bert_classifier_all
#SBATCH --mail-type=END
#SBATCH --mail-user=cer446@nyu.edu
#SBATCH --output=bert_classifier_all_base_sl50_bs16_lr2e-5_ep3.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module purge
module load python3/intel/3.6.3
source ~/pyenv-BERT/py3.6.3/bin/activate

export msl=50
export model=bert-base-uncased

python run_classifier_mod.py \
  --label Obscenity \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir DATA/Obscenity/ \
  --bert_model $model \
  --max_seq_length $msl \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/output1/

python run_classifier_mod.py \
  --label hatespeech \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir DATA/hatespeech/ \
  --bert_model $model \
  --max_seq_length $msl \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/output2/

python run_classifier_mod.py \
  --label namecalling \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir DATA/namecalling/ \
  --bert_model $model \
  --max_seq_length $msl \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/output3/

python run_classifier_mod.py \
  --label negprejudice \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir DATA/negprejudice/ \
  --bert_model $model \
  --max_seq_length $msl \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/output4/

python run_classifier_mod.py \
  --label noneng \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir DATA/noneng/ \
  --bert_model $model \
  --max_seq_length $msl \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/output5/

python run_classifier_mod.py \
  --label porn \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir DATA/porn/ \
  --bert_model $model \
  --max_seq_length $msl \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/output6/

python run_classifier_mod.py \
  --label stereotypes \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir DATA/stereotypes/ \
  --bert_model $model \
  --max_seq_length $msl \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/output7/

python run_classifier_mod.py \
  --label Threat \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir DATA/Threat/ \
  --bert_model $model \
  --max_seq_length $msl \
  --train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/output8/