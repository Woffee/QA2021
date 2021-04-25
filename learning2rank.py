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

def generate_ltr_data(questions, documents, args, to_file, ltr_train_ns_mount=1, candidates_file=''):
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

                for i in range(ltr_train_ns_mount):
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
        candidates = {}
        if candidates_file != '' and os.path.exists(candidates_file):
            print("===loading candidates...")
            logger.info("===loading candidates...")
            with open(candidates_file, "r") as f:
                for line in f:
                    l = line.strip()
                    if l != "":
                        arr = l.split(" ")
                        qid = arr[0]
                        dids = arr[1:]
                        candidates[qid] = dids

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

                if q.id in candidates.keys():
                    candidate_apis = candidates[q.id]
                    for did in candidate_apis:
                        doc = documents_dict[did]
                        q_encoder_input.append(q.matrix)
                        r_decoder_input.append(doc.matrix)
                        weight_data_r.append(doc.weight)

                        qid_list.append(q.id)
                        did_list.append(doc.id)

                        if did in q.answer_ids:
                            y_data.append(1)
                            y_list.append(1)
                        else:
                            y_data.append(0)
                            y_list.append(0)
                else:
                    print("=== generate test ltr data from all documents")
                    logger.info("=== generate test ltr data from all documents")
                    for doc in documents:
                        q_encoder_input.append(q.matrix)
                        r_decoder_input.append(doc.matrix)
                        weight_data_r.append(doc.weight)

                        qid_list.append(q.id)
                        did_list.append(doc.id)

                        if doc.id in q.answer_ids:
                            y_data.append(1)
                            y_list.append(1)
                        else:
                            y_data.append(0)
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

    parser.add_argument('--candidates_file', help='log_file', type=str, default='data/QA2021_stackoverflow4_candidates.txt')

    parser.add_argument('--ltr_train_file', help='ltr_train_file', type=str, default="for_ltr/ltr_train.txt")
    parser.add_argument('--ltr_vali_file', help='ltr_vali_file', type=str, default="for_ltr/ltr_vali.txt")
    parser.add_argument('--ltr_eval_file', help='ltr_eval_file', type=str, default="for_ltr/ltr_eval.txt")

    parser.add_argument('--ltr_qrel_file', help='ltr_qrel_file', type=str, default="data/QA2021_qrel.txt")
    parser.add_argument('--ltr_pred_file', help='ltr_pred_file', type=str, default="data/QA2021_ltr_pred.txt")

    parser.add_argument('--weight_path', help='weight_path', type=str, default='ckpt/best_model_qa6_epo40_neg2_bat32_back.hdf5')
    parser.add_argument('--output_length', help='output_length', type=int, default=1000)

    args = parser.parse_args()
    print(args)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("training parameters %s", args)

    ltr_train_file = args.ltr_train_file
    ltr_vali_file = args.ltr_vali_file
    ltr_eval_file = args.ltr_eval_file

    to_pred_file = args.ltr_pred_file
    to_file_qrel = args.ltr_qrel_file

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
        generate_ltr_data(questions[train_num: train_num+vali_num], documents, args, ltr_vali_file, -1, args.candidates_file)
        generate_ltr_data(questions[train_num+vali_num:], documents, args, ltr_eval_file, -1, args.candidates_file)

        # Evaluation:
        if not os.path.exists(to_file_qrel):
            with open(to_file_qrel, "w") as fw:
                for q in questions[train_num+vali_num:]:
                    for doc_id in set(q.answer_ids):
                        fw.write("%s 0 %s 1\n" % (q.id, doc_id))
            print("saved to %s" % to_file_qrel)


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


    with open(to_pred_file, "w") as fw:
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
    print("save to: %s" % to_pred_file)
    print('NDCG, Random ranking:', metric.calc_mean_random(Eqids, Ey))
    print('NDCG, Our model:', metric.calc_mean(Eqids, Ey, Epred))
    print("done")





