import dataclasses
from typing import Dict, Optional, Union

from lm_eval.utils import eval_logger

from rouge_score import rouge_scorer

default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# matching DSKD's evaluation
def rouge(prediction, ground_truth):
    scorer = default_rouge_scorer
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def process_results(doc, results):
    return {
        "rougeL": rouge(results[0], doc["output"]),
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc
