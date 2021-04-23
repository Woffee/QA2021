"""


@Time    : 10/30/20
@Author  : Wenbo
"""

import os
import time
import math
import logging
import argparse
import pyltr
import random
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import adam
from keras.layers.recurrent import GRU
from keras.layers.core import Lambda
from keras.layers import Dot, add, Bidirectional, Dropout, Reshape, Concatenate, Dense, MaxPooling1D, Flatten, Masking
from keras.models import Input, Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from adding_weight import adding_weight

from main import Question, Document, loss_c, negative_samples, no_negative_samples, read_questions, read_docs, get_stemmed_words, init_doc_matrix, preprocess_all_questions, preprocess_all_documents, train_w2v

logger = None
random.seed(2021)

def generate_ltr_data(questions, documents, args, to_file, ltr_ns_mount=1):
    logger.info("=== extract_features...")
    ns_amount = args.ns_amount

    documents_dict = {}
    documents_for_train = []  # 选取数据集里出现过的 documents，用作训练。因为大部分 API 几乎很少被用到。
    for doc in documents:
        doc_id = doc.id
        documents_dict[doc_id] = doc
        if doc.count > 0:
            documents_for_train.append(doc)

    input_length = questions[0].matrix.shape[0]
    output_length = documents[0].matrix.shape[0]
    print("=== input_length: %d, output_length: %d" % (input_length, output_length))
    logger.info("=== input_length: %d, output_length: %d" % (input_length, output_length))

    model = negative_samples(input_length=input_length,
                             input_dim=args.input_dim,
                             output_length=output_length,
                             output_dim=args.output_dim,
                             hidden_dim=args.hidden_dim,
                             ns_amount=ns_amount,
                             learning_rate=args.learning_rate,
                             drop_rate=args.drop_rate)
    model.load_weights(args.weight_path)

    # 这个 model 只做预测
    new_nn_model = no_negative_samples(input_length=input_length,
                                       input_dim=args.input_dim,
                                       output_length=output_length,
                                       output_dim=args.output_dim,
                                       hidden_dim=args.hidden_dim,
                                       ns_amount=ns_amount,
                                       learning_rate=args.learning_rate,
                                       drop_rate=args.drop_rate)
    # print(new_nn_model.summary())
    for i in range(len(model.layers)):
        weights = model.layers[i].get_weights()
        new_nn_model.layers[i].set_weights(weights)

    # 提取 features
    # https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
    extractor = keras.Model(inputs=new_nn_model.inputs,
                            outputs=[new_nn_model.get_layer('dropout1').output, new_nn_model.get_layer('dropout2').output])


    if to_file.find("train") >= 0:
        batch = 100 # 一次运行数量，防止内存不够
        steps = math.ceil( len(questions) / batch)
        for step in range(steps):
            start = step * batch
            end = start + batch
            end = min(end, len(questions))

            q_encoder_input = []
            r_decoder_input = []
            weight_data_r = []
            y_data = []

            qid_list = []
            did_list = []
            y_list = []

            for ii in range(start, end):
                q = questions[ii]
                print("now: %d, qid:%s" % (ii, q.id) )

                for did in q.answer_ids:
                    doc = documents_dict[did]
                    q_encoder_input.append(q.matrix)
                    r_decoder_input.append(doc.matrix)
                    weight_data_r.append(doc.weight)
                    y_data.append(1)

                    qid_list.append(q.id)
                    did_list.append(doc.id)
                    y_list.append(1)

                for i in range(ltr_ns_mount):
                    r = random.randint(1, len(documents_for_train) - 1)
                    while (documents_for_train[r].id in q.answer_ids):
                        r = random.randint(1, len(documents_for_train) - 1)

                    doc = documents_for_train[r]
                    q_encoder_input.append(q.matrix)
                    r_decoder_input.append(doc.matrix)
                    weight_data_r.append(doc.weight)
                    y_data.append(0)

                    qid_list.append(q.id)
                    did_list.append(doc.id)
                    y_list.append(0)

            # features = extractor([q_encoder_input, r_decoder_input, weight_data_r])
            features = extractor.predict([q_encoder_input, r_decoder_input, weight_data_r])

            with open(to_file, "a") as f:
                for j in range(len(q_encoder_input)):
                    row1 = features[0][j]
                    row2 = features[1][j]

                    feature_str = ''
                    for k in range(len(row1)):
                        feature_str = feature_str + (" %d:%.9f" % (k + 1, row1[k]))
                    for k in range(len(row2)):
                        feature_str = feature_str + (" %d:%.9f" % (k + 1 + len(row1), row2[k]))

                    label = y_list[j]
                    doc_id = did_list[j]
                    qid = qid_list[j]

                    line = "%d qid:%s%s # %s \n" % (label, qid, feature_str,doc_id)
                    f.write(line)
            print("saved to: %s" % to_file)
    else: # vali or eval
        batch = 100  # 一次运行数量，防止内存不够
        steps = math.ceil(len(questions) / batch)
        for step in range(steps):
            start = step * batch
            end = start + batch
            end = min(end, len(questions))

            q_encoder_input = []
            r_decoder_input = []
            weight_data_r = []
            y_data = []

            qid_list = []
            did_list = []
            y_list = []

            for ii in range(start, end):
                q = questions[ii]
                print("now: %d, qid:%s" % (ii, q.id))

                for doc in documents_for_train:
                    if doc.id in q.answer_ids:
                        y_data.append(1)
                        y_list.append(1)
                    else:
                        y_data.append(0)
                        y_list.append(0)


                    q_encoder_input.append(q.matrix)
                    r_decoder_input.append(doc.matrix)
                    weight_data_r.append(doc.weight)

                    qid_list.append(q.id)
                    did_list.append(doc.id)

            # features = extractor([q_encoder_input, r_decoder_input, weight_data_r])
            features = extractor.predict([q_encoder_input, r_decoder_input, weight_data_r])

            with open(to_file, "a") as f:
                for j in range(len(q_encoder_input)):
                    row1 = features[0][j]
                    row2 = features[1][j]

                    feature_str = ''
                    for k in range(len(row1)):
                        feature_str = feature_str + (" %d:%.9f" % (k + 1, row1[k]))
                    for k in range(len(row2)):
                        feature_str = feature_str + (" %d:%.9f" % (k + 1 + len(row1), row2[k]))

                    label = y_list[j]
                    doc_id = did_list[j]
                    qid = qid_list[j]

                    line = "%d qid:%s%s # %s \n" % (label, qid, feature_str, doc_id)
                    f.write(line)
            print("saved to: %s" % to_file)
        pass


if __name__ == '__main__':
    default_log_file = time.strftime("%Y-%m-%d", time.localtime()) + '.log'

    parser = argparse.ArgumentParser(description='Test for argparse')

    parser.add_argument('--input_dim', help='input_dim', type=int, default=100)
    parser.add_argument('--output_dim', help='output_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', help='hidden_dim', type=int, default=64)
    parser.add_argument('--ns_amount', help='ns_amount', type=int, default=10)
    parser.add_argument('--doc_mode', help='doc_mode', type=int, default=0)  # 0:for all,  1:for those apprear on the SO

    parser.add_argument("--pool_s", default=20, type=int, help="")
    parser.add_argument("--pool_stride", default=5, type=int, help="")

    parser.add_argument('--learning_rate', help='learning_rate', type=float, default=0.001)
    parser.add_argument('--drop_rate', help='drop_rate', type=float, default=0.01)

    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--epochs', help='epochs', type=int, default=100)


    parser.add_argument('--data_type', help='data_type', type=str, default='stackoverflow4')
    parser.add_argument('--qa_file', help='qa_file', type=str, default='stackoverflow4/QA_list.txt')
    parser.add_argument('--doc_file', help='doc_file', type=str, default='stackoverflow4/Doc_list.txt')
    parser.add_argument('--log_file', help='log_file', type=str, default=default_log_file)

    parser.add_argument('--weight_path', help='weight_path', type=str, default='ckpt/best_model_qa6_epo40_neg2_bat32_back.hdf5')

    args = parser.parse_args()
    print(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("training parameters %s", args)

    qa_version = "qa6"
    ltr_train_file = "for_ltr/ltr_%s_%s_train.txt" % (args.data_type, qa_version)
    ltr_vali_file = "for_ltr/ltr_%s_%s_vali.txt" % (args.data_type, qa_version)
    ltr_eval_file = "for_ltr/ltr_%s_%s_eval.txt" % (args.data_type, qa_version)
    to_file = "data/pyltr_%s_%s_pred.txt" % (args.data_type, qa_version)

    if not os.path.exists(ltr_train_file):
        question_answers = read_questions(args.qa_file)
        documents = read_docs(args.doc_file)

        train_num = int(len(question_answers) * 0.8)
        vali_num =  int(len(question_answers) * 0.1)
        eval_num =  int(len(question_answers) * 0.1)
        print("total: %d, train: %d" % (len(question_answers), train_num))

        w2v_path = "models/w2v_%s.bin" % args.data_type
        w2v = train_w2v(question_answers[:train_num], documents, w2v_path)

        questions = preprocess_all_questions(question_answers, w2v)
        documents = preprocess_all_documents(question_answers, documents, w2v)

        generate_ltr_data(questions[:train_num], documents, args, ltr_train_file, 10)
        generate_ltr_data(questions[train_num: train_num+vali_num], documents, args, ltr_vali_file, 2)
        generate_ltr_data(questions[train_num+vali_num:], documents, args, ltr_eval_file, 2)

        # Evaluation:
        to_file_qrel = "data/QA2021_%s_qrel.txt" % args.data_type
        if not os.path.exists(to_file_qrel):
            with open(to_file_qrel, "w") as fw:
                for q in questions[train_num+vali_num:]:
                    for doc_id in set(q.answer_ids):
                        fw.write("%s 0 %s 1\n" % (q.id, doc_id))
            print("saved to %s" % to_file_qrel)


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

    print("training learning2rank...")
    logger.info("training learning2rank...")
    with open(ltr_train_file) as trainfile, \
            open(ltr_vali_file) as valifile, \
            open(ltr_eval_file) as evalfile:
        TX, Ty, Tqids, Tcomments = pyltr.data.letor.read_dataset(trainfile)
        VX, Vy, Vqids, Vcomments = pyltr.data.letor.read_dataset(valifile)
        EX, Ey, Eqids, Ecomments = pyltr.data.letor.read_dataset(evalfile)

    metric = pyltr.metrics.NDCG(k=10)

    # Only needed if you want to perform validation (early stopping & trimming)
    monitor = pyltr.models.monitors.ValidationMonitor(
        VX, Vy, Vqids, metric=metric, stop_after=250)

    model = pyltr.models.LambdaMART(
        metric=metric,
        n_estimators=500,
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


    with open(to_file, "w") as fw:
        for qid in res_data.keys():
            rank_list = res_data[qid]
            rank_list = sorted(rank_list, key=lambda item: item[0], reverse=True)

            ii = 1
            visited = []
            for score, api in rank_list:
                if api not in visited:
                    visited.append(api)
                    fw.write("%s\tQ0\t%s\t%d\t%.8f\t%s\n" % (qid, api, ii, score, "indri"))
                    ii += 1
    print("save to: %s" % to_file)
    print('NDCG, Random ranking:', metric.calc_mean_random(Eqids, Ey))
    print('NDCG, Our model:', metric.calc_mean(Eqids, Ey, Epred))
    print("done")





