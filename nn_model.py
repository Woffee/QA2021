"""


@Time    : 4/4/21
@Author  : Wenbo
"""

import os
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

import keras
import argparse

import time
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
now_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

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
def negative_samples(input_length, input_dim, output_length, output_dim, hidden_dim, ns_amount, learning_rate, drop_rate):
    q_encoder_input = Input(shape=(input_length, input_dim))
    r_decoder_input = Input(shape=(output_length, output_dim))
    weight_data_r = Input(shape=(1,))
    weight_data_w = Input(shape=(1, ns_amount))
    weight_data_w_list = Lambda(lambda x: tf.split(x, num_or_size_splits=ns_amount, axis=2))(weight_data_w)
    fixed_r_decoder_input = adding_weight(output_length, output_dim)([r_decoder_input, weight_data_r])
    w_decoder_input = Input(shape=(output_length, output_dim, ns_amount))
    w_decoder_input_list = Lambda(lambda x: tf.split(x, num_or_size_splits=ns_amount, axis=3))(w_decoder_input)
    fixed_w_decoder_input = []
    for i in range(ns_amount):
        w_decoder_input_list[i] = Reshape((output_length, output_dim))(w_decoder_input_list[i])
        weight_data_w_list[i] = Reshape((1,))(weight_data_w_list[i])
        fixed_w_decoder_input.append(adding_weight(output_length, output_dim)([w_decoder_input_list[i], weight_data_w_list[i]]))

    encoder = Bidirectional(GRU(hidden_dim), merge_mode="ave", name="bidirectional1")
    q_encoder_output = encoder(q_encoder_input)
    q_encoder_output = Dropout(rate=drop_rate, name="dropout1")(q_encoder_output)

    decoder = Bidirectional(GRU(hidden_dim), merge_mode="ave", name="bidirectional2")
    r_decoder_output = decoder(fixed_r_decoder_input)
    r_decoder_output = Dropout(rate=drop_rate, name="dropout2")(r_decoder_output)

    # doc_output = MaxPooling1D(pool_size=20, stride=5, padding='same')(q_encoder_input)
    # doc_output = Flatten()(q_encoder_input)
    # que_output = MaxPooling1D(pool_size=20, stride=5, padding='same')(fixed_r_decoder_input)
    # que_output = Flatten()(fixed_r_decoder_input)

    output_vec = Concatenate(axis=1, name="dropout_con")([q_encoder_output, r_decoder_output])
    output_hid = Dense(hidden_dim, name="output_hid")(output_vec)
    similarity = Dense(1, name="similarity")(output_hid)

    # Difference between kernel, bias, and activity regulizers in Keras
    # https://stats.stackexchange.com/questions/383310/difference-between-kernel-bias-and-activity-regulizers-in-keras
    # output = Dense(128, kernel_regularizer=keras.regularizers.l2(0.0001))(output_vec) # activation="relu",
    # output = Dense(64, name="output_hid", kernel_regularizer=keras.regularizers.l2(0.0001))(output) # activation="relu",
    # similarity = Dense(1, name="similarity", activation="softmax")(output)

    w_decoder_output_list = []
    for i in range(ns_amount):
        w_decoder_output = decoder(fixed_w_decoder_input[i])
        w_decoder_output = Dropout(rate=drop_rate)(w_decoder_output)
        w_decoder_output_list.append(w_decoder_output)
    similarities = [ similarity ]
    # similarities = [Dot(axes=1, normalize=True)([q_encoder_output, r_decoder_output])]
    for i in range(ns_amount):
        similarities.append(Dot(axes=1, normalize=True)([q_encoder_output, w_decoder_output_list[i]]))
    loss_data = Lambda(lambda x: loss_c(x))(similarities)
    model = Model([q_encoder_input, r_decoder_input, w_decoder_input, weight_data_r, weight_data_w], similarities[0])
    ada = adam(lr=learning_rate)
    model.compile(optimizer=ada, loss=lambda y_true, y_pred: loss_data)
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


def train(w2v_model, qa_file, doc_file, to_model_file, to_ckpt_file, args):
    logger.info("preprocessing...")
    ns_amount = args.ns_amount

    questions = []
    answers = []

    # 计算每个question的向量
    input_length = 0
    with open(qa_file, "r", encoding='utf-8') as f:
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
                        ans.append(int(a) - 1) # 因为原始数据从1开始计数，这里减去1。改为从0开始。
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
    print("len(doc_vecs)",len(doc_vecs))
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


    # [q_encoder_input, r_decoder_input, w_decoder_input, weight_data_r, weight_data_w]
    q_encoder_input = []
    r_decoder_input = []
    w_decoder_input = []
    weight_data_r = []
    weight_data_w = []
    y_data = []

    total = len(question_vecs)

    # 打乱训练数据
    qa_index = list( range(total) )
    # random.shuffle(qa_index)

    for i in qa_index:
        y = [1] + [0] * ns_amount
        y_data.append(y)
        # question
        q_encoder_input.append( question_vecs[i] )
        # 每个question一个正确答案
        aid = answers[i][0]
        r_decoder_input.append( doc_vecs[ aid ])
        weight_data_r.append(doc_weight[ aid ])
        # 10个un-related答案
        aids = get_randoms(list(doc_weight.keys()), [aid], 10)
        w_decoder = []
        w_weight = []
        for aid in aids:
            w_decoder.append( doc_vecs[aid] )
            w_weight.append( doc_weight[ aid ])
        w_decoder = np.array(w_decoder).reshape(output_length, args.input_dim, ns_amount)
        w_weight = np.array(w_weight).reshape((1, ns_amount))
        w_decoder_input.append(w_decoder)
        weight_data_w.append(w_weight)
    y_data = np.array(y_data).reshape(total, (1+ns_amount))

    # w_decoder_input = np.array(w_decoder_input)
    # print(w_decoder_input.shape)
    # reshape(ns_amount, output_length, 200)
    train_num = int(total * 0.9)
    model = negative_samples(input_length=input_length,
                             input_dim=args.input_dim,
                             output_length=output_length,
                             output_dim=args.output_dim,
                             hidden_dim=args.hidden_dim,
                             ns_amount=ns_amount,
                             learning_rate=args.learning_rate,
                             drop_rate=args.drop_rate)
    print(model.summary())

    print("start training...")
    logger.info("start training...")
    model.fit([q_encoder_input[:train_num], r_decoder_input[:train_num], w_decoder_input[:train_num], weight_data_r[:train_num], weight_data_w[:train_num] ], y_data[:train_num],
              batch_size=args.batch_size,
              epochs=args.epochs,
              verbose=1,
              validation_data=([q_encoder_input[train_num:], r_decoder_input[train_num:], w_decoder_input[train_num:], weight_data_r[train_num:], weight_data_w[train_num:] ], y_data[train_num:])
              )

    res = model.evaluate([q_encoder_input[train_num:], r_decoder_input[train_num:], w_decoder_input[train_num:], weight_data_r[train_num:], weight_data_w[train_num:] ], y_data[train_num:],verbose=1)
    print("training over.")
    logger.info("training over")
    print(model.metrics_names)
    print(res)
    print(model.summary())


    model.save(to_model_file)
    print("saved model to:", to_model_file)

    model.save_weights(to_ckpt_file)
    print("saved weights to:", to_ckpt_file)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--data_type', help='data_type',  type=str, default='twitter')

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

    parser.add_argument('--qa_file', help='qa_file', type=str, default='')
    parser.add_argument('--doc_file', help='doc_file', type=str, default='')
    parser.add_argument('--model_path', help='model_path', type=str, default='')
    parser.add_argument('--weight_path', help='weight_path', type=str, default='')

    parser.add_argument('--log_file', help='log_file', type=str, default='')


    parser.add_argument('--output_length', help='output_length', type=int, default=1000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log_file)
    logger = logging.getLogger(__name__)



    logger.info("training parameters %s", args)

    data_type = args.data_type

    print("running model.py, data_type: %s" % data_type)
    logger.info("running model.py, data_type: %s" % data_type)



    path = "models/%s.wv.cbow.d%d.w10.n10.bin" % (data_type, args.input_dim)
    to_model_file = args.model_path
    to_ckpt_file = args.weight_path


    w2v_model = KeyedVectors.load_word2vec_format(path, binary=True)

    qa_path = args.qa_file
    doc_path = args.doc_file

    # res = sentence2vec(w2v_model, "I want to determine Quantity sold", 10)
    train(w2v_model, qa_path, doc_path, to_model_file, to_ckpt_file, args)

