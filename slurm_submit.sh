#!/bin/bash
#
#SBATCH -c 1 # number of cores
#SBATCH --gres=gpu:1 # number of gpus
#SBATCH --mem=16000 # memory pool for all cores
#SBATCH -t 6-12:00 # time (D-HH:MM)
#SBATCH -o slurm_logs/out/slurm_consti_top_roberta_7_12.%N.%j.out # STDOUT
#SBATCH -e slurm_logs/err/slurm_consti_top_roberta_7_12.%N.%j.err # STDERR

nvidia-smi

declare -a ALL_TASKS
ALL_TASKS+=( "srl-conll" )
echo "All tasks to run: ${ALL_TASKS[@]}"
for task in "${ALL_TASKS[@]}"
do
    # Usage: probing.sh -m <model_name> -c <config_file> [-t <single_task>] 
    srun probing.sh -m roberta-base -t "$task" -c probing/jiant/config/edgeprobe/edgeprobe_roberta.conf
done