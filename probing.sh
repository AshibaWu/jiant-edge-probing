#!/bin/bash
# Master script to start a full suite of edge probing experiments. 
#
# Usage:
#  ./run_exp.sh -p <project_name>
#
function bert_mix_k_exp() {
    # Run BERT with ELMo-style scalar mixing across the first K layers.
    # Usage: bert_mix_k_exp <task_name> <bert_model_name> <k>
    OVERRIDES="exp_name=bert-${2}-mix_${3}-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=bert-$2"
    OVERRIDES+=", transformers_output_mode=mix"
    OVERRIDES+=", transformers_max_layer=${3}"
    run_exp "jiant/config/edgeprobe/edgeprobe_bert.conf" "${OVERRIDES}"
}

function bert_at_k_exp() {
    # Run BERT and probe layer K.
    # Usage: bert_at_k_exp <task_name> <bert_model_name> <k>
    OVERRIDES="exp_name=bert-${2}-at_${3}-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=bert-$2"
    OVERRIDES+=", transformers_output_mode=top"
    OVERRIDES+=", transformers_max_layer=${3}"
    run_exp "jiant/config/edgeprobe/edgeprobe_bert.conf" "${OVERRIDES}"
}

set -e

# Default arguments.
PROJECT_NAME=""


##
# All experiments below.
# Uncomment the lines you want to run, or comment out those you don't.
##

declare -a ALL_TASKS
ALL_TASKS+=( "srl" )
echo "All tasks to run: ${ALL_TASKS[@]}"

##
# Experiments for the ACL paper ("BERT layer paper"), comparing the different
# layers of BERT.
export GPU_TYPE="p100"
for task in "${ALL_TASKS[@]}"
do
    # Probe BERT-base
    for k in $(seq -f "%02.f" 0 12); do
        kuberun bert-base-uncased-at-${k}-$task   "bert_at_k_exp  edges-$task base-uncased ${k}"
        kuberun bert-base-uncased-mix-${k}-$task  "bert_mix_k_exp edges-$task base-uncased ${k}"
    done
done
export GPU_TYPE="v100"
for task in "${ALL_TASKS[@]}"
do
    # Probe BERT-large
    for k in $(seq-f "%02.f" 0 24); do
        kuberun bert-large-uncased-at-${k}-$task   "bert_at_k_exp  edges-$task large-uncased ${k}"
        kuberun bert-large-uncased-mix-${k}-$task  "bert_mix_k_exp edges-$task large-uncased ${k}"
    done
done
