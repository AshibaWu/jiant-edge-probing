#!/usr/bin/env python

# Convenience script to score a set of runs from the predictions.
#
# Usage:
#  python get_all_scores.py -i /path/to/experiments/*/run \
#      -o /tmp/sample.tsv [--parallel n]
#
# Output will be a json file containing aggregated and per-class
# predictions for each run.


import argparse
import collections
import json
import logging as log
import os
import re
import sys
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

import numpy as np

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def find_tasks_and_splits(path: str) -> List[Tuple[str, str]]:
    """Find tasks and splits for a particular run."""
    matches = []
    for fdir in os.listdir(path):
        if "tagged" in fdir and ("ner" in fdir or "pos" in fdir):
            for fname in os.listdir(path + "/" + fdir):
                if fname == "results.tsv":
                    matches.append(fdir)
    if not matches:
        log.warning("Warning: no predictions found for run '%s'", path)
    return matches

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="inputs", type=str, nargs="+", help="Input files.")
    args = parser.parse_args(args)
    print("args:", args)
    aggregated_result = {}
    """ 
        aggregated_result = {
            "{end_task_name}-{model}-{layer}": {
                (idx): {
                "text": "why are african - americans so beautiful ?",
                "idx": 0,
                "label": "neutral",
                "pos_accuracy": 0.9,
                "ner_accuracy": None,
                "pos": [{
                    "label": "WRB",
                    "span1": [0, 1],
                    "mention": "Why",
                    "pred": "WRB",
                    "proba": [0.23, 0.01, ...]
                    },
                    { ... }],
                "pos_correct_cnt": 2,
                "ner": [],
                ...
            },
            {...},
            ...
        }
    """
    work_items = []
    # aggregate results from all lower level tasks
    for path in args.inputs:
        for task in find_tasks_and_splits(path):
            print("Task:", task)
            work_items.append((path, task))
            # Global run info from log file.task_info = exp_dir.split("_")
            task_info = task.split("_")
            if len(task_info) != 2:
                log.warning("Warning: task name format is incorrect")
            sample = {}
            sample["model"] = "-".join(task_info[0].split("-")[:-1])
            sample["layer"] = task_info[1].split("-")[0]
            sample["method"] = task_info[0].split("-")[-1]
            sample["probing_task"] = "-".join(task_info[1].split("-")[1:])
            if task_info[1].split("-")[-1] == 'ner' \
                or task_info[1].split("-")[-1] == 'pos':
                sample["sub_task_name"] = task_info[1].split("-")[-1]
                sample["end_task"] = task_info[1].split("-")[-2]
            elif task_info[1].split("-")[-2] == 'dep' \
                or task_info[1].split("-")[-2] == 'srl':
                sample["sub_task_name"] = task_info[1].split("-")[-2]
                sample["end_task"] = task_info[1].split("-")[-1]
            else:
                break

            system = f'{sample["end_task"]}_{sample["model"]}_{sample["layer"]}'
            
            test_pred_file = os.path.join(path, task, 'run', sample["probing_task"] + "_test.json")
            if not os.path.isfile(test_pred_file):
                log.warning("Warning: no predictions for '%s'", sample["probing_task"])
                return None
                
            if system not in aggregated_result:
                aggregated_result[system] = {}
            
            label_file = os.path.join(path, task, 'vocab', sample["probing_task"] + "_labels.txt")
            labels = []
            with open(label_file) as f:
                for line in f:
                    labels.append(line.strip())
            print(f"processing {test_pred_file}")
            with open(test_pred_file) as f:
                for line in tqdm(f):
                    preds = json.loads(line)
                    if preds["info"]["idx"] not in aggregated_result[system]:
                        aggregated_result[system][preds["info"]["idx"]] = {
                            "idx": preds["info"]["idx"],
                        }
                    if "field" not in preds["info"]:
                        aggregated_result[system][preds["info"]["idx"]]["text"] = preds["text"]
                    else:
                        aggregated_result[system][preds["info"]["idx"]][preds["info"]["field"]] = preds["text"]

                    aggregated_result[system][preds["info"]["idx"]][sample["sub_task_name"]+"_correct_cnt"] = 0
                    aggregated_result[system][preds["info"]["idx"]][sample["sub_task_name"]+"_cnt"] = 0
                    for target in preds["targets"]:
                        target["pred"] = labels[np.argmax(target["preds"]["proba"])]
                        if target["pred"] == target["label"]:
                            aggregated_result[system][preds["info"]["idx"]][sample["sub_task_name"]+"_correct_cnt"] += 1
                        aggregated_result[system][preds["info"]["idx"]][sample["sub_task_name"]+"_cnt"] += 1
        
        
    # generate metrics such as accuracy
    supported_subtasks = ['ner', 'pos', 'srl', 'dep']
    for system, samples in aggregated_result.items():
        with open('results/'+ system + '.json', 'w') as fp:
            for sample in samples.values():
                for st in supported_subtasks:
                    if st + "_cnt" not in sample.keys():
                        sample[st + "_cnt"] = 0
                        sample[st + "_correct_cnt"] = 0
                        sample[st + "_acc"] = None
                    else:
                        sample[st + "_acc"] = sample[st + "_correct_cnt"] / sample[st + "_cnt"]
                        
                fp.write(json.dumps(sample))
                fp.write("\n")
        with open('results/'+ system + '.json', 'w') as fp:
            fp.write(json.dumps(samples))
    


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
