"""
读取训练好的 Model 的参数，生成 QA representations。

然后训练 learning2rank model，并做预测。

@Time    : 4/4/21
@Author  : Wenbo
"""


import os
import io
import sys
import numpy as np

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import adam
from keras.layers.recurrent import GRU
from keras.layers.core import Lambda
from keras.layers import Dot, add, Bidirectional, Dropout, Reshape, Concatenate, Dense, MaxPooling1D, Flatten
from keras.models import Input, Model
from keras import backend as K
from adding_weight import adding_weight
from nn_model import negative_samples

import keras
import argparse

import time
import logging

from word2vec import remove_punc
import nltk
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
nltk.download('punkt')

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

import random
random.seed(10)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



def loss_c(similarity):
    ns_num = len(similarity) - 1
    if ns_num < 1:
        print("There needs to have at least one negative sample!")
        exit()
    else:
        loss_amount = K.exp(-1 * add([similarity[0], -1*similarity[1]]))
        for i in range(ns_num - 1):
            loss_amount = add([loss_amount, K.exp(-1 * add([similarity[0], -1*similarity[i + 2]]))])
        loss_amount = K.log(1 + loss_amount)
        return loss_amount


#input_length: How many words in one questions (MAX)
#input_dim: How long the representation vector for one word for questions
#output_length: How many words in one document (MAX)
#output_dim: How long the representation vector for one word for documents
#hidden_dim: Hidden size for network
#ns_amount: Negative samples amount
#learning_rate: Learning rate for the model
#drop_out_rate: Drop out rate when doing the tarining
#q_encoder_input: Question (batch_size, input_length, input_dim)
#r_decoder_input: Related API document (when doing prediction, it is the document you want to check the relationship score)(batch_size, output_length, output_dim)
#e.g. for question Q, if you want to check the relationship score between Q and document D, then you put D here.
#w_decoder_input: Unrelated API documents (when doing prediction, it can be input with zero array which will not influence result)(batch_size, output_length, output_dim, ns_amount)
#weight_data_1: Weight (Ti/Tmax) for related document(batch_size, 1)
#weight_data_2: Weights (Ti/Tmax) for unrelated documents(batch_size, 1, ns_amount)
def no_negative_samples(input_length, input_dim, output_length, output_dim, hidden_dim, ns_amount, learning_rate, drop_rate):
    q_encoder_input = Input(shape=(input_length, input_dim))
    r_decoder_input = Input(shape=(output_length, output_dim))
    weight_data_r = Input(shape=(1,))

    fixed_r_decoder_input = adding_weight(output_length, output_dim)([r_decoder_input, weight_data_r])

    encoder = Bidirectional(GRU(hidden_dim), merge_mode="ave", name="bidirectional1")
    q_encoder_output = encoder(q_encoder_input)
    q_encoder_output = Dropout(rate=drop_rate, name="dropout1")(q_encoder_output)

    decoder = Bidirectional(GRU(hidden_dim), merge_mode="ave", name="bidirectional2")
    r_decoder_output = decoder(fixed_r_decoder_input)
    r_decoder_output = Dropout(rate=drop_rate, name="dropout2")(r_decoder_output)

    output_vec = Concatenate(axis=1, name="dropout_con")([q_encoder_output, r_decoder_output])
    output_hid = Dense(hidden_dim, name="output_hid")(output_vec)
    similarity = Dense(1, name="similarity")(output_hid)

    # Difference between kernel, bias, and activity regulizers in Keras
    # https://stats.stackexchange.com/questions/383310/difference-between-kernel-bias-and-activity-regulizers-in-keras
    # output = Dense(128, kernel_regularizer=keras.regularizers.l2(0.0001))(output_vec) # activation="relu",
    # output = Dense(64, name="output_hid", kernel_regularizer=keras.regularizers.l2(0.0001))(output) # activation="relu",
    # similarity = Dense(1, name="similarity", activation="softmax")(output)

    model = Model([q_encoder_input, r_decoder_input, weight_data_r], similarity)
    ada = adam(lr=learning_rate)
    model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=[keras.metrics.categorical_accuracy])
    return model



def sentence2vec(w2v_model, s, max_length):
    if isinstance(s, str):
        words = word_tokenize( remove_punc( s.lower() ) )
    else:
        words = s
    vec = []
    if len(words) > max_length:
        words = words[:max_length]
    for word in words:
        if word in w2v_model.wv.vocab:
            vec.append(w2v_model.wv[word])
    dim = len(vec[0])
    # print("dim", dim)
    print("len(vec)",len(vec))
    for i in range(max_length - len(vec)):
        vec.append( np.zeros(dim) )
    return np.array(vec)


def get_randoms(arr, not_in, num=2):
    ma = len(arr)
    res = []
    for i in range(num):
        r = random.randint(1, ma-1)
        while( arr[r] in not_in ):
            r = random.randint(1, ma-1)
        res.append(arr[r])
    return res


def get_train_data(data_type, w2v_model,  qa_file, doc_file, to_file_path, args):
    logger.info("preprocessing...")
    ns_amount = args.ns_amount

    questions = []
    answers = []

    # 计算每个question的向量
    input_length = 0
    with open(qa_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i >= 2000:
                break
            line = line.strip().lower()
            if line != "" and i % 2 == 0:
                words = word_tokenize(remove_punc(line))
                input_length = max(len(words), input_length)
                questions.append(words)
            elif line != "" and i % 2 == 1:
                arr = line.strip().split(" ")
                ans = []
                for a in arr:
                    if a != "":
                        ans.append(int(a) - 1)  # 因为原始数据从1开始计数，这里减去1。改为从0开始。
                answers.append(ans)

    question_vecs = []
    for q_words in questions:
        question_vecs.append(sentence2vec(w2v_model, q_words, input_length))
    print("len(question_vecs)", len(question_vecs))

    # 计算每个document的向量
    docs = []
    output_length = 0
    with open(doc_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line != "":
                words = word_tokenize(remove_punc(line))
                output_length = max(len(words), output_length)
                docs.append(words)
    doc_vecs = []
    output_length = args.output_length
    for d_words in docs:
        doc_vecs.append(sentence2vec(w2v_model, d_words, output_length))
    print("len(doc_vecs)", len(doc_vecs))
    logger.info("input_length:%d, output_length:%d" % (input_length, output_length))

    # 计算每个doc出现的频率
    doc_count = {}
    for ans in answers:
        for a in ans:
            if a in doc_count.keys():
                doc_count[a] += 1
            else:
                doc_count[a] = 1

    # 计算每个doc的weight
    doc_weight = {}
    t_max = 0
    for k in doc_count.keys():
        t_max = max(t_max, doc_count[k])
    for k in doc_count.keys():
        doc_weight[k] = doc_count[k] / t_max

    total = len(question_vecs)
    train_num = int(total * 0.9)
    logger.info("train_num:%d, total:%d" % (train_num, total))

    # 打乱数据
    qa_index = list(range(total))
    # random.shuffle(qa_index)

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
        # print("layer %d: " % i, weights)


    # 这个 model 用于提取 QA representations
    qa_vec_model = Model(inputs=new_nn_model.input, outputs=new_nn_model.get_layer('output_hid').output)

    step = 0
    while step * 200 <= train_num:
        # [q_encoder_input, r_decoder_input, w_decoder_input, weight_data_r, weight_data_w]
        q_encoder_input = []
        r_decoder_input = []
        weight_data_r = []
        y_data = []

        qid_list = []
        label_list = []
        aid_list = []

        logger.info("step: %d" % step)

        end = min(train_num, (step + 1) * 200)
        for ss in range(step * 200, end):
            i = qa_index[ss]
            logger.info("question: %d" % i)
            qid_list.append(i)
            label_list.append(1)

            y = 1
            y_data.append(y)
            # question
            q_encoder_input.append(question_vecs[i])
            # 每个question一个正确答案
            aid = answers[i][0]
            aid_list.append(aid)
            r_decoder_input.append(doc_vecs[aid])
            weight_data_r.append(doc_weight[aid])
            # 10个un-related答案
            aids = get_randoms(list(doc_weight.keys()), [aid], 10)
            for aaid in aids:
                qid_list.append(i)
                label_list.append(0)
                aid_list.append(aaid)

                # 这些答案都是unrelated
                y = 0
                y_data.append(y)
                # question
                q_encoder_input.append(question_vecs[i])
                r_decoder_input.append(doc_vecs[aaid])
                weight_data_r.append(doc_weight[aaid])

        logger.info("predicting...")
        res = qa_vec_model.predict([q_encoder_input, r_decoder_input, weight_data_r])

        with open(to_file_path, "a") as f:
            for i in range(len(res)):
                row = res[i]
                feature_str = ''
                for j in range(len(row)):
                    feature_str = feature_str + (" %d:%.9f" % (j + 1, row[j]))
                label = label_list[i]
                id = qid_list[i]
                doc_id = aid_list[i]

                line = "%d qid:%d%s # doc-%d \n" % (label, id, feature_str, doc_id)
                f.write(line)
        print("saved to:", to_file_path)
        logger.info("step:%d added" % step)
        step += 1

    logger.info("saved to: %s" % to_file_path)


def get_test_data(data_type, w2v_model,  qa_file, doc_file, to_file_path, args):
    logger.info("preprocessing...")
    ns_amount = args.ns_amount

    questions = []
    answers = []

    # 计算每个question的向量
    input_length = 0
    with open(qa_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i >= 2000:
                break
            line = line.strip().lower()
            if line != "" and i % 2 == 0:
                words = word_tokenize(remove_punc(line))
                input_length = max(len(words), input_length)
                questions.append(words)
            elif line != "" and i % 2 == 1:
                arr = line.strip().split(" ")
                ans = []
                for a in arr:
                    if a != "":
                        ans.append(int(a) - 1)  # 因为原始数据从1开始计数，这里减去1。改为从0开始。
                answers.append(ans)

    question_vecs = []
    for q_words in questions:
        question_vecs.append(sentence2vec(w2v_model, q_words, input_length))
    print("len(question_vecs)", len(question_vecs))

    # 计算每个document的向量
    docs = []
    output_length = 0
    with open(doc_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line != "":
                words = word_tokenize(remove_punc(line))
                output_length = max(len(words), output_length)
                docs.append(words)
    doc_vecs = []
    output_length = args.output_length
    for d_words in docs:
        doc_vecs.append(sentence2vec(w2v_model, d_words, output_length))
    print("len(doc_vecs)", len(doc_vecs))
    logger.info("input_length:%d, output_length:%d" % (input_length, output_length))

    # 计算每个doc出现的频率
    doc_count = {}
    for ans in answers:
        for a in ans:
            if a in doc_count.keys():
                doc_count[a] += 1
            else:
                doc_count[a] = 1

    # 计算每个doc的weight
    doc_weight = {}
    t_max = 0
    for k in doc_count.keys():
        t_max = max(t_max, doc_count[k])
    for k in doc_count.keys():
        doc_weight[k] = doc_count[k] / t_max

    total = len(question_vecs)
    train_num = int(total * 0.9)
    logger.info("train_num:%d, total:%d" % (train_num, total))

    # 打乱数据
    qa_index = list(range(total))
    # random.shuffle(qa_index)

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
        # print("layer %d: " % i, weights)


    # 这个 model 用于提取 QA representations
    qa_vec_model = Model(inputs=new_nn_model.input, outputs=new_nn_model.get_layer('output_hid').output)


    for ss in range( train_num, total):
        i = qa_index[ss]

        # [q_encoder_input, r_decoder_input, w_decoder_input, weight_data_r, weight_data_w]
        q_encoder_input = []
        r_decoder_input = []
        weight_data_r = []

        logger.info("now %d, get all documents for question: %d" % (ss,i) )
        print("now %d, get all documents for question: %d" % (ss,i))

        # qid_list.append(i)
        # label_list.append(1)

        cur_answers = answers[i]
        doc_list_ordered = [a for a in cur_answers]
        for aid in list(doc_weight.keys()):
            if aid not in doc_list_ordered:
                doc_list_ordered.append(aid)

        label_list = []
        aid_list = []

        print("len(doc_list_ordered):", len(doc_list_ordered))
        print("len(cur_answers):", len(cur_answers))

        for aid in doc_list_ordered:
            aid_list.append(aid)
            if aid in cur_answers:
                label_list.append(1)
            else:
                label_list.append(0)

            # question
            q_encoder_input.append(question_vecs[i])
            r_decoder_input.append(doc_vecs[aid])
            weight_data_r.append(doc_weight[aid])

        logger.info("now:%d , predicting question: %d" % (ss,i))
        print("now:%d , predicting question: %d" % (ss,i))

        start = 0
        end = len(q_encoder_input)
        for cur in range(0, end, 1000):
            print("cur:%d / %d" % (cur, end))
            a = q_encoder_input[cur:cur+1000]
            b = r_decoder_input[cur:cur+1000]
            d = weight_data_r[cur:cur+1000]


            res = qa_vec_model.predict([a,b,d])
            # print(res)

            with open(to_file_path, "a") as f:
                for j in range(len(res)):
                    row = res[j]
                    feature_str = ''
                    for k in range(len(row)):
                        feature_str = feature_str + (" %d:%.9f" % (k + 1, row[k]))
                    label = label_list[j]
                    doc_id = aid_list[j]

                    line = "%d qid:%d%s # doc-%d \n" % (label, i, feature_str,doc_id)
                    f.write(line)
    print("saved to:", to_file_path)
    logger.info("total:%d" % total)
    logger.info("saved to: %s" % to_file_path)

def get_qrel_data(args):
    if os.path.exists(args.to_qrel_file):
        return

    NUM = 200

    # questions and doc-ids
    qid_list = []
    truth_id_list = []
    lines = io.open(args.qa_file, encoding='UTF-8').read().strip().split('\n')
    for i, line in enumerate(lines):
        if i >= 2000:
            break

        print("now:", i)
        l = line.strip().lower()
        if l != "":
            if i % 2 == 0:
                question = l
                qid = i / 2
                qid_list.append(qid)
            else:
                # query-id 0 document-id relevance
                truth_ids = [int(id) for id in l.split(" ")]
                truth_id_list.append(truth_ids)


    total = len(qid_list)
    train_num = int(total * 0.9)
    logger.info("train_num:%d, total:%d" % (train_num, total))

    with open(args.to_qrel_file, "w") as fw:
        for ss in range(train_num, total):
            qid = qid_list[ss]
            truth_ids = truth_id_list[ss]
            for did in truth_ids:
                fw.write("%d 0 doc-%d 1\n" % (qid, did-1))
            for j in range(NUM):
                rand_id = j + 1
                if rand_id not in truth_ids:
                    truth_ids.append(rand_id)
                    fw.write("%d 0 doc-%d 0\n" % (qid, rand_id-1))

    print("saved to", args.to_qrel_file)

if __name__ == '__main__':
    now_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())


    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--data_type', help='data_type', type=str, default='twitter')

    parser.add_argument('--input_dim', help='input_dim', type=int, default=200)
    parser.add_argument('--output_dim', help='output_dim', type=int, default=200)
    parser.add_argument('--hidden_dim', help='hidden_dim', type=int, default=64)
    parser.add_argument('--ns_amount', help='ns_amount', type=int, default=10)

    parser.add_argument("--pool_s", default=20, type=int, help="")
    parser.add_argument("--pool_stride", default=5, type=int, help="")

    parser.add_argument('--learning_rate', help='learning_rate', type=float, default=0.001)
    parser.add_argument('--drop_rate', help='drop_rate', type=float, default=0.01)

    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--epochs', help='epochs', type=int, default=20)

    parser.add_argument('--output_length', help='output_length', type=int, default=1000)
    parser.add_argument('--log_file', help='output_length', type=str, default=BASE_DIR + '/' + now_time + '.log')

    parser.add_argument('--qa_file', help='qa_file', type=str, default='')
    parser.add_argument('--doc_file', help='doc_file', type=str, default='')
    parser.add_argument('--model_path', help='model_path', type=str, default='')
    parser.add_argument('--weight_path', help='weight_path', type=str, default='')
    parser.add_argument('--to_train_file', help='to_train_file', type=str, default='')
    parser.add_argument('--to_test_file', help='to_test_file', type=str, default='')
    parser.add_argument('--to_qrel_file', help='to_qrel_file', type=str, default='')
    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log_file)
    logger = logging.getLogger(__name__)
    logger.info("=== training parameters %s", args)

    w2v_path = "models/%s.wv.cbow.d%d.w10.n10.bin" % (args.data_type, args.input_dim)
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)


    get_train_data(args.data_type, w2v_model, args.qa_file, args.doc_file, args.to_train_file, args)
    get_test_data(args.data_type, w2v_model, args.qa_file, args.doc_file, args.to_test_file, args)
    get_qrel_data(args)

