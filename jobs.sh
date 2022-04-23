#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --mem=16000
source ~/.bashrc
export JIANT_PROJECT_PREFIX=/home/yunxuan2/Probing/jiant-edge-probing
export JIANT_DATA_DIR=probing/data
export WORD_EMBS_FILE=emb/crawl-300d-2MMM.vec
conda activate jiant

python main.py --config_file probing/jiant/config/edgeprobe/edgeprobe_bert.conf   -o "target_tasks=edges-srl-conll,exp_name=ep_bert_srl_all_mix,input_module=roberta-base,transformers_output_mode=mix,transformers_max_layer=4"

