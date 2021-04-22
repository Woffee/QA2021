"""


@Time    : 10/30/20
@Author  : Wenbo
"""

import os
import time
import argparse
import pyltr

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Test for argparse')
    # parser.add_argument('--data_type', help='data_type', type=str, default='twitter')
    # parser.add_argument('--model_type', help='model_type', type=str, default='lambdaMART')
    # parser.add_argument('--ranklib_path', help="ranklib_path", type=str, default='')
    #
    # parser.add_argument('--train_file', help="train_file", type=str, default='')
    # parser.add_argument('--test_file', help="test_file", type=str, default='')
    #
    #
    # parser.add_argument('--train', action='store_true', help="train learning2rank model")
    # parser.add_argument('--pred', action='store_true', help="do predict")
    #
    # args = parser.parse_args()
    #
    # print(args)
    #
    # data_type = args.data_type
    # print("data_type:", data_type)
    #
    # # ranklib_path = '/Users/woffee/www/gis_technical_support/gis_qa/ltrdemo2wenbo/utils/bin/RankLib.jar'
    #
    # # /Users/woffee/www/gis_qa/ltrdemo2wenbo/evaluation.py
    # # trec_eval_path= '/Users/woffee/www/trec_eval-9.0.7/trec_eval'
    #
    # m_dict = {
    #     'RankNet': 1,
    #     'lambdaMART': 6
    # }
    #
    # train_metric = 'MAP'
    # train_model = args.model_type
    #
    # data_train_path = args.train_file
    # data_test_path = args.test_file
    #
    # save_model_path = BASE_DIR + '/ltr/models/' + data_type + "_" + train_model + '_' + train_metric + ".txt"
    # data_pred_path = BASE_DIR + '/ltr/predicts/' + data_type + "_" + train_model + '_' + train_metric + "_pred.txt"
    #
    # # train
    # ranker = m_dict[train_model]
    # train_cmd = "java -jar %s -train %s -ranker %d -tvs 0.8 -metric2t MAP -save %s" % (args.ranklib_path, data_train_path, ranker, save_model_path)
    #
    # # pred
    # pred_cmd = "java -jar %s -rank %s -load %s -indri %s" % (args.ranklib_path, data_test_path, save_model_path, data_pred_path)
    #
    # if args.train:
    #     print(train_cmd)
    #     os.system(train_cmd)
    # if args.pred:
    #     print(pred_cmd)
    #     os.system(pred_cmd)

    with open('for_ltr/data-train.txt') as trainfile, \
            open('for_ltr/data-test.txt') as valifile, \
            open('for_ltr/data-test.txt') as evalfile:
        TX, Ty, Tqids, Tcomments = pyltr.data.letor.read_dataset(trainfile)
        VX, Vy, Vqids, Vcomments = pyltr.data.letor.read_dataset(valifile)
        EX, Ey, Eqids, Ecomments = pyltr.data.letor.read_dataset(evalfile)

    metric = pyltr.metrics.NDCG(k=10)

    # Only needed if you want to perform validation (early stopping & trimming)
    monitor = pyltr.models.monitors.ValidationMonitor(
        VX, Vy, Vqids, metric=metric, stop_after=250)

    model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=1000,
        learning_rate=0.02,
        max_features=0.5,
        query_subsample=0.5,
        max_leaf_nodes=10,
        min_samples_leaf=64,
        verbose=1,
    )

    model.fit(TX, Ty, Tqids, monitor=monitor)

    Epred = model.predict(EX)

    doc_list = list(Ecomments)
    qid_list = list(Eqids)

    res_data = {}
    for i, pred in enumerate(list(Epred)):
        qid = qid_list[i]
        did = doc_list[i]

        if qid in res_data.keys():
            res_data[qid].append( [pred, did] )
        else:
            res_data[qid] = [ [pred, did] ]

    to_file = "data/pyltr_pred.txt"
    with open(to_file, "w") as fw:
        for qid in qid_list:
            rank_list = res_data[qid]
            rank_list = sorted(rank_list, key=lambda item: item[0], reverse=True)

            ii = 1
            for score, api in rank_list:
                fw.write("%s\tQ0\t%s\t%d\t%.8f\t%s\n" % (qid, api, ii, score, "indri"))
                ii += 1
    print("save to: %s" % to_file)
    print('NDCG, Random ranking:', metric.calc_mean_random(Eqids, Ey))
    print('NDCG, Our model:', metric.calc_mean(Eqids, Ey, Epred))
    print("done")





