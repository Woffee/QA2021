"""


@Time    : 10/30/20
@Author  : Wenbo
"""

import os
import time
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--data_type', help='data_type', type=str, default='twitter')
    parser.add_argument('--model_type', help='model_type', type=str, default='lambdaMART')
    parser.add_argument('--ranklib_path', help="ranklib_path", type=str, default='')

    parser.add_argument('--train_file', help="train_file", type=str, default='')
    parser.add_argument('--test_file', help="test_file", type=str, default='')


    parser.add_argument('--train', action='store_true', help="train learning2rank model")
    parser.add_argument('--pred', action='store_true', help="do predict")

    args = parser.parse_args()

    print(args)

    data_type = args.data_type
    print("data_type:", data_type)

    # ranklib_path = '/Users/woffee/www/gis_technical_support/gis_qa/ltrdemo2wenbo/utils/bin/RankLib.jar'

    # /Users/woffee/www/gis_qa/ltrdemo2wenbo/evaluation.py
    # trec_eval_path= '/Users/woffee/www/trec_eval-9.0.7/trec_eval'

    m_dict = {
        'RankNet': 1,
        'lambdaMART': 6
    }

    train_metric = 'MAP'
    train_model = args.model_type

    data_train_path = args.train_file
    data_test_path = args.test_file

    save_model_path = BASE_DIR + '/ltr/models/' + data_type + "_" + train_model + '_' + train_metric + ".txt"
    data_pred_path = BASE_DIR + '/ltr/predicts/' + data_type + "_" + train_model + '_' + train_metric + "_pred.txt"

    # train
    ranker = m_dict[train_model]
    train_cmd = "java -jar %s -train %s -ranker %d -tvs 0.8 -metric2t MAP -save %s" % (args.ranklib_path, data_train_path, ranker, save_model_path)

    # pred
    pred_cmd = "java -jar %s -rank %s -load %s -indri %s" % (args.ranklib_path, data_test_path, save_model_path, data_pred_path)

    if args.train:
        print(train_cmd)
        os.system(train_cmd)
    if args.pred:
        print(pred_cmd)
        os.system(pred_cmd)

    print("done")





