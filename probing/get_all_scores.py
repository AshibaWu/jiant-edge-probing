#!/usr/bin/env python

# Convenience script to collect all subtask scores.
#
# Usage:
#  python get_all_scores.py -i /path/to/experiments/*/run \
#      -o /tmp/stats.tsv [--parallel n]
#
# Output will be a long-form TSV file containing aggregated and per-class
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

import analysis
from data import utils
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats

log.basicConfig(format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO)


def find_tasks_and_splits(run_path: str) -> List[Tuple[str, str]]:
    """Find tasks and splits for a particular run."""
    matches = []
    for fdir in os.listdir(run_path):
        if "tagged" in fdir and "ner" in fdir:
            for fname in os.listdir(run_path + "/" + fdir):
                if fname == "results.tsv":
                    matches.append(fdir)
    if not matches:
        log.warning("Warning: no predictions found for run '%s'", run_path)
    return matches


def get_result_info(all_dir: str, exp_dir: str, result_name="results.tsv") -> pd.DataFrame:
    """Extract some run information from the log text."""
    result_path = os.path.join(all_dir, exp_dir, result_name)
    if not os.path.isfile(result_path):
        log.warning("Warning: no results.tsv found for run '%s'", result_path)
        return None

    results = {}
    with open(result_path) as fd:
        first_line = fd.readlines()[0].rstrip()
        results = first_line.split('\t')[1]
        results = dict(map(lambda x: (x.split(': ')[0].split('_')[1] ,x.split(': ')[1]), results.split(', ')))
    task_info = exp_dir.split("_")
    if len(task_info) != 2:
        log.warning("Warning: task name format is incorrect")
    stats = pd.DataFrame([results], columns=results.keys())
    stats["model"] = "-".join(task_info[0].split("-")[:-1])
    stats["layer"] = task_info[1].split("-")[0]
    stats["method"] = task_info[0].split("-")[-1]
    stats["end_task"] = task_info[1].split("-")[-2]
    stats["probing_task"] = task_info[1].split("-")[-1]
    print(stats)
    return stats

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="inputs", type=str, nargs="+", help="Input files.")
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of runs to process in parallel."
    )
    args = parser.parse_args(args)
    print("args:", args)

    all_results = pd.DataFrame()
    for run_path in args.inputs:
        for task in find_tasks_and_splits(run_path):
            # Global run info from log file.
            all_results = pd.concat([all_results, get_result_info(run_path, task)], ignore_index=True)
    print(all_results)
    all_results[['model', 'layer', 'method', 'end_task', 'probing_task', 'acc', 'precision', 'recall', 'f1']].to_csv('tmp/results.tsv', sep='\t', index=False)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
