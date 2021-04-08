"""
MAP, MRR

AUC, Accuracy

NDCG, Precision, Recall @ 5, 10, 15, 20



Approximately simulates trec_eval using pytrec_eval.

https://github.com/cvangysel/pytrec_eval/blob/master/examples/trec_eval.py

Demo output:

args.qrel: /Users/woffee/www/emse-apiqa/reviewer_baselines/word2api/data/word2api_stackoverflow3_qrel.txt
args.run /Users/woffee/www/emse-apiqa/reviewer_baselines/word2api/data/word2api_stackoverflow3_pred.txt
==========
P_10                     all     0.0421
P_15                     all     0.0331
P_20                     all     0.0273
P_5                      all     0.0606
map                      all     0.1788
ndcg                     all     0.2565
recall_10                all     0.3518
recall_15                all     0.4148
recall_20                all     0.4509
recall_5                 all     0.2553
recip_rank               all     0.1999


@Time    : 4/7/21
@Author  : Wenbo
"""


import argparse
import os
import sys

import pytrec_eval


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--qrel', default='/Users/woffee/www/emse-apiqa/reviewer_baselines/word2api/data/word2api_stackoverflow3_qrel.txt')
    parser.add_argument('--run', default='/Users/woffee/www/emse-apiqa/reviewer_baselines/word2api/data/word2api_stackoverflow3_pred.txt')

    args = parser.parse_args()

    print("args.qrel:", args.qrel)
    print("args.run", args.run)

    assert os.path.exists(args.qrel)
    assert os.path.exists(args.run)

    with open(args.qrel, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(args.run, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, pytrec_eval.supported_measures)

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    total = len(results.items())
    sum_map = 0.0

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            # print_line(measure, query_id, value)
            pass

    # Scope hack: use query_measures of last item in previous loop to
    # figure out all unique measure names.
    #
    # TODO(cvangysel): add member to RelevanceEvaluator
    #                  with a list of measure names.
    print("==========")
    selected_measures = ['map', 'recip_rank', 'P_5', 'P_10', 'P_15', 'P_20', 'recall_5', 'recall_10', 'recall_15', 'recall_20','ndcg']
    for measure in sorted(query_measures.keys()):
        if measure in selected_measures:
            print_line(
                measure,
                'all',
                pytrec_eval.compute_aggregated_measure(
                    measure,
                    [query_measures[measure]
                     for query_measures in results.values()]))

if __name__ == "__main__":
    sys.exit(main())
