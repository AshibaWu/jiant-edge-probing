# Reproduce Edge Probing

This repository can reproduce the results for papers:
- What do you learn from context? Probing for sentence structure in contextualized word representations ("edge probing")
- BERT Rediscovers the Classical NLP Pipeline ("BERT layer paper")

The original edge probing repo [jiant-v1-legacy](https://github.com/nyu-mll/jiant-v1-legacy) provides lots of useful information, but after jiant migrated from v1 to v2, I went through some extra steps to rerun the scripts.
Here are the revised steps you should follow:

## Env setup
Set up conda environment
```conda env create -f environment.yml```
Activate the environment
```
conda activate jiant
export JIANT_PROJECT_PREFIX=/home/wangchew/jiant-v1-legacy
export JIANT_DATA_DIR=probing/data
export WORD_EMBS_FILE=emb/crawl-300d-2MMM.vec
```
WORD_EMBS_FILE is just a placeholder since we don't need this for bert experiments

## (Optional) Download and run with test data
Prepare for test data. Scripts are modified to only process ud data and bert related part.
```
bash get_and_process_all_data.sh
```
You should now have probing/data/edges/dep_ewt folder to test

## If you already have data in hand
Tokenize it with retokenize_edge_data.py
```
python probing/retokenize_edge_data.py -t roberta-base probing/data/edges/conll_srl/test.json
```

## Automated scripts to run combination of trainings:
Run using slurm cluster
```
sbatch slurm_submit.sh
```
Run using local machine. You can specify the config file and model
```
bash probing.sh -c probing/jiant/config/edgeprobe/edgeprobe_roberta.conf -m roberta-base
```

## 

# Parameters & Training
- transformers_output_mode: there is two mode: "top", and "mix"
- transformers_max_layer: Maximum layer to return from BERT encoder. Layer 0 is wordpiece embeddings. bert_embeddings_mode will behave as if the BERT encoder is truncated at this layer, so 'top' will return this layer, and 'mix' will return a mix of all layers up to and including this layer. Set to -1 to use all layers. Used for probing experiments.
```
python main.py --config_file probing/jiant/config/edgeprobe/edgeprobe_bert.conf   -o "target_tasks=edges-,exp_name=ep_bert_demo,input_module=bert-base-uncased"
```
```
python main.py --config_file probing/jiant/config/edgeprobe/edgeprobe_bert.conf   -o "target_tasks=edges-srl-conll,exp_name=ep_bert_srl_all_mix,input_module=bert-base-cased,transformers_output_mode=mix"
```
Train on mix of layer 1-4
```
python main.py --config_file probing/jiant/config/edgeprobe/edgeprobe_bert.conf   -o "target_tasks=edges-srl-conll,exp_name=ep_bert_srl_all_mix,input_module=bert-base-cased,transformers_output_mode=mix,transformers_max_layer=4"
```

### Only evaluate on the task
```
python main.py --config_file probing/jiant/config/edgeprobe/edgeprobe_bert.conf   -o "target_tasks=edges-srl-conll,exp_name=ep_bert_srl_conll_all_top,input_module=bert-base-cased,transformers_output_mode=top,do_target_task_training=0,do_pretrain=0,do_full_eval=1"
```

### Generate probing metrics
```
./scripts/edgeprobing/analyze_project.sh /path/to/project
# Expected directory structure:
# <project_dir>/
#   <experiment_1>/
#     run/
#     vocab/
#   <experiment_2>/
#     run/
#     vocab/
#   (...)
```
# Edge Probing Config
In the paper, they train the span pooling and MLP classifiers to extract information from the contextual vectors. Here is the settings from jiant-v1-legacy/jiant/config/defaults.conf

Add more tasks here

```
// Edge-Probing Experiments //
// See README for context.

// Template: Not used for any single task, but extended per-task below.
edges-tmpl {
    span_classifier_loss_fn = "sigmoid"  // 'sigmoid' or 'softmax'
    classifier_span_pooling = "attn"  // 'attn' or 'x,y'
    classifier_hid_dim = 256
    classifier_dropout = 0.3
    pair_attn = 0

    // Default iters; run 50k steps.
    max_vals = 100
    val_interval = 500
}
edges-tmpl-small = ${edges-tmpl} {
    // 10k steps
    val_interval = 100
}
edges-tmpl-large = ${edges-tmpl} {
    // 250k steps
    max_vals = 250
    val_interval = 1000
}

edges-pos-ontonotes = ${edges-tmpl-large}
edges-nonterminal-ontonotes = ${edges-tmpl-large}
edges-ner-ontonotes = ${edges-tmpl-large}
edges-srl-ontonotes = ${edges-tmpl-large}
edges-coref-ontonotes = ${edges-tmpl-large}

edges-dep-ud-ewt = ${edges-tmpl-large}

edges-spr1 = ${edges-tmpl-small}
edges-spr2 = ${edges-tmpl-small}
edges-dpr = ${edges-tmpl-small}

edges-rel-semeval = ${edges-tmpl-small}
edges-rel-tacred = ${edges-tmpl}

// new edge probing tasks
edges-srl-conll = ${edges-tmpl-large}
```
# Edge Probing Details
## Change Probing layer
jiant-v1-legacy/jiant/huggingface_transformers_interface/modules.py
```
available_layers = hidden_states[: self.max_layer + 1]
```

## Change Probing Classifier (Head)
jiant-v1-legacy/probing/jiant/modules/edge_probing.py
```
self.classifier = Classifier.from_params(clf_input_dim, task.n_classes, task_params)
```
## Experiments setup of original paper
```
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
```

# Citation
```bibtex
@inproceedings{47786,
title	= {What do you learn from context? Probing for sentence structure in contextualized word representations},
author	= {Ian Tenney and Patrick Xia and Berlin Chen and Alex Wang and Adam Poliak and R. Thomas McCoy and Najoung Kim and Benjamin Van Durme and Samuel R. Bowman and Dipanjan Das and Ellie Pavlick},
year	= {2019},
URL	= {https://openreview.net/forum?id=SJzSgnRcKX},
booktitle	= {International Conference on Learning Representations}
}

@article{tenney2019bert,
  title={BERT rediscovers the classical NLP pipeline},
  author={Tenney, Ian and Das, Dipanjan and Pavlick, Ellie},
  journal={arXiv preprint arXiv:1905.05950},
  year={2019}
}

```


