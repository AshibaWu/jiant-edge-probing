#!/bin/bash
# Master script to start a full suite of edge probing experiments. 
#
# Usage:
#  ./probing.sh -m <model_name> -c <config_file> [-t <single_task>]  
#
echo $@
export JIANT_PROJECT_PREFIX=/projects/tir6/general/probing/jiant-edge-probing/experiments # absolute path
export JIANT_DATA_DIR=probing/data # relative
export WORD_EMBS_FILE=emb/crawl-300d-2MMM.vec
eval "$(conda shell.bash hook)"
conda activate jiant
DEFAULT_MODEL_NAME="bert-base-uncased"
DEFAULT_CONFIG_FILE="jiant/config/edgeprobe/edgeprobe_bert.conf"

# Handle flags.
OPTIND=1         # Reset in case getopts has been used previously in the shell.
while getopts ":m:c:t:" opt; do
    case "$opt" in
    m)  MODEL_NAME=$OPTARG
        ;;
    c)  CONFIG_FILE=$OPTARG
        ;;
    t)  TASK=$OPTARG
        ;;
    \? )
        echo "Invalid flag $opt."
        exit 1
        ;;
    esac
done
shift $((OPTIND-1))

# Remaining positional arguments.
MODE=${1:-"create"}

model_name="${MODEL_NAME:-$DEFAULT_MODEL_NAME}" # bert-base-cased, bert-base-uncased, roberta-base
config_file="${CONFIG_FILE:-$DEFAULT_CONFIG_FILE}"

if [ -z $model_name ]; then
    echo "You must provide a model name!"
    exit 1
fi

# use the tasks name specified in 
declare -a ALL_TASKS
ALL_TASKS+=( "$TASK" )
echo "All tasks to run: ${ALL_TASKS[@]}"


function run_exp() {
    # Helper function to invoke main.py.
    # Don't run this directly - use the experiment functions below,
    # or create a new one for a new experiment suite.
    # Usage: run_exp <config_file> <overrides>
    OVERRIDES=$1
    declare -a args
    args+=( --config_file "${config_file}" )
    args+=( -o "${OVERRIDES}" )
    if [ ! -z $NOTIFY_EMAIL ]; then
        args+=( --notify "$NOTIFY_EMAIL" )
    fi
    echo "python main.py ${args[@]}"
    python main.py "${args[@]}"
}

function bert_mix_k_exp() {
    # Run BERT with ELMo-style scalar mixing across the first K layers.
    # Usage: bert_mix_k_exp <task_name> <bert_model_name> <k>
    OVERRIDES="exp_name=${2}-mix_${3}-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=$2"
    OVERRIDES+=", transformers_output_mode=mix"
    OVERRIDES+=", transformers_max_layer=${3}"
    run_exp "${OVERRIDES}"
}

function bert_at_k_exp() {
    # Run BERT and probe layer K.
    # Usage: bert_at_k_exp <task_name> <bert_model_name> <k>
    OVERRIDES="exp_name=${2}-at_${3}-${1}, run_name=run"
    OVERRIDES+=", target_tasks=$1"
    OVERRIDES+=", input_module=$2"
    OVERRIDES+=", transformers_output_mode=top"
    OVERRIDES+=", transformers_max_layer=${3}"
    run_exp "${OVERRIDES}"
}

set -e

# Default arguments.
PROJECT_NAME=""
# Set the desired probing layer here
layers=$(seq -f "%02.f" 12 1 12)
echo "config file: $config_file"
echo "probe model: $model_name"
echo $(IFS=,; echo "probing layers: ${layers[*]}")
echo "============ Start experiments ============"

##
# Change the model: {roberta-uncased, base-uncased}
# layer 06, 12 = seq -f "%02.f" 6 6 12

for task in "${ALL_TASKS[@]}"
do
    # Probe
    for k in $layers; do
        # bert_at_k_exp  edges-$task ${model_name} ${k}
        bert_mix_k_exp edges-$task ${model_name} ${k}
    done
done
