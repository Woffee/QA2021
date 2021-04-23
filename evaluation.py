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
import numpy as np
import math
import pytrec_eval
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix

def min_max(arr):
    mi = np.min(arr)
    ma = np.max(arr)
    if mi == ma:
        return [1] * len(arr)
    res = []
    for x in arr:
        res.append( float(x - mi) / (ma - mi) )
    return res


def calc_auc(qrel_file, pred_file):
    # read ground truth file
    q_answers = {}
    with open(qrel_file, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line == "":
                continue

            qid, Q0, did, relevant = line.split(" ")

            if relevant == "1":
                if qid in q_answers.keys():
                    q_answers[qid].append(did)
                else:
                    q_answers[qid] = [did]

    # read pred file
    q_pred = {}
    with open(pred_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == "":
                continue

            arr = line.split("\t")
            qid = arr[0].strip()
            did = arr[2].strip()
            rank = int(arr[3].strip())
            score = float(arr[4].strip())
            if qid in q_pred.keys():
                q_pred[qid].append([did, score])
            else:
                q_pred[qid] = [[did, score]]
            # print("did:",did,", score:", score)

    # auc & accuracy
    auc_list = []
    accuracy_list = []


    for qid in q_pred.keys():
        y_true = []
        y_score = []
        ans = q_answers[qid]
        for did, score in q_pred[qid]:
            if did in ans:
                y_true.append(1)
            else:
                y_true.append(0)

            y_score.append(score)

        if np.sum(y_true) == 0:
            accuracy_list.append(0)
            auc_list.append(0)
            print("No positive samples in y_true, qid:%s" % qid)
            continue

        y_score = min_max(y_score)
        y_score = np.array(y_score)

        y_pred = []
        for s in y_score:
            if s >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)

        y_true = np.array(y_true)

        # 下面两种方式计算auc结果相同。但方式二能返回更多信息。

        # 方式 1
        # auc = roc_auc_score(y_true, y_score)

        # 方式 2
        try:
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        except Exception as e:
            pass
        auc = metrics.auc(fpr, tpr)
        if math.isnan(auc):
            auc = 0

        auc_list.append(auc)
        # print("qid: %s  -  auc:%.4f" % (qid, auc))
        # if qid == "1":
        #     pass

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy_list.append(accuracy)

    final_auc = np.mean(auc_list)
    final_accuracy = np.mean(accuracy_list)
    print("=== AUC Evaluation results ===")
    print("Test total:", len(auc_list))
    print("Qrel file:", qrel_file)
    print("Pred file:", pred_file)
    # print("len(pred.keys):", len(q_pred.keys()))
    print("AUC      =", final_auc)
    print("Accuracy =", final_accuracy)
    return final_auc, final_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--qrel', default='/Users/woffee/www/emse-apiqa/QA2021/data/QA2021_stackoverflow4_qrel.txt')
    parser.add_argument('--run', default='/Users/woffee/www/emse-apiqa/QA2021/data/pyltr_pred.txt')

    args = parser.parse_args()

    print("args.qrel:", args.qrel)
    print("args.run", args.run)

    assert os.path.exists(args.qrel)
    assert os.path.exists(args.run)

    final_auc, final_accuracy = calc_auc(args.qrel, args.run)

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

    eva_values = {}
    for measure in selected_measures:
        eva_values[measure] = pytrec_eval.compute_aggregated_measure( measure,
                [query_measures[measure] for query_measures in results.values()])
            # print_line( measure, 'all', eva_values[measure])
    for measure in selected_measures:
        print_line( measure, 'all', eva_values[measure])


    print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" %
          (final_auc, final_accuracy, eva_values['map'], eva_values['recip_rank'],
           eva_values['P_5'], eva_values['P_10'], eva_values['P_15'], eva_values['P_20'],
           eva_values['recall_5'], eva_values['recall_10'], eva_values['recall_15'], eva_values['recall_20'],
           eva_values['ndcg']))


if __name__ == "__main__":
    sys.exit(main())
