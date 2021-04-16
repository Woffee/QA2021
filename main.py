
import os
import sys
import numpy as np
import multiprocessing
from gensim.models import Word2Vec

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import adam
from keras.layers.recurrent import GRU
from keras.layers.core import Lambda
from keras.layers import Dot, add, Bidirectional, Dropout, Reshape, Concatenate, Dense, MaxPooling1D, Flatten, Masking
from keras.models import Input, Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

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
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import SnowballStemmer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

import random
random.seed(10)

import string
printable = set(string.printable)



class Question:
    def __init__(self, id, question_text, answer_ids):
        self.id = id
        self.question_text = question_text
        self.answer_ids = answer_ids

        self.words = None # stemmed words in the question_text
        self.matrix = None # embeddings of words

class Document:
    def __init__(self, id, doc_text):
        self.id = id
        self.doc_text = doc_text

        self.words = None # stemmed words in the doc_text
        self.matrix = None # embeddings of words
        self.weight = 0
        self.count = 0 # number of occurrences in the data set


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

        w_decoder_weighted = adding_weight(output_length, output_dim)([w_decoder_input_list[i], weight_data_w_list[i]])
        w_decoder_weighted_masked = Masking(mask_value=0., input_shape=(output_length, output_dim))(w_decoder_weighted)
        fixed_w_decoder_input.append(w_decoder_weighted_masked)

    q_encoder_input_masked = Masking(mask_value=0., input_shape=(input_length, input_dim) )(q_encoder_input)
    fixed_r_decoder_input_masked = Masking(mask_value=0., input_shape=(output_length, output_dim) )(fixed_r_decoder_input)

    encoder = Bidirectional(GRU(hidden_dim), merge_mode="ave", name="bidirectional1")
    q_encoder_output = encoder(q_encoder_input_masked)
    q_encoder_output = Dropout(rate=drop_rate, name="dropout1")(q_encoder_output)

    decoder = Bidirectional(GRU(hidden_dim), merge_mode="ave", name="bidirectional2")
    r_decoder_output = decoder(fixed_r_decoder_input_masked)
    r_decoder_output = Dropout(rate=drop_rate, name="dropout2")(r_decoder_output)

    # doc_output = MaxPooling1D(pool_size=20, stride=5, padding='same')(q_encoder_input)
    # doc_output = Flatten()(q_encoder_input)
    # que_output = MaxPooling1D(pool_size=20, stride=5, padding='same')(fixed_r_decoder_input)
    # que_output = Flatten()(fixed_r_decoder_input)

    output_vec = Concatenate(axis=1, name="dropout_con")([q_encoder_output, r_decoder_output])
    output_hid = Dense(hidden_dim, name="output_hid")(output_vec)
    similarity = Dense(1, name="similarity", activation="softmax")(output_hid)

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

def no_negative_samples(input_length, input_dim, output_length, output_dim, hidden_dim, ns_amount, learning_rate, drop_rate):
    q_encoder_input = Input(shape=(input_length, input_dim))
    r_decoder_input = Input(shape=(output_length, output_dim))
    weight_data_r = Input(shape=(1,))

    fixed_r_decoder_input = adding_weight(output_length, output_dim)([r_decoder_input, weight_data_r])

    q_encoder_input_masked = Masking(mask_value=0., input_shape=(input_length, input_dim))(q_encoder_input)
    fixed_r_decoder_input_masked = Masking(mask_value=0., input_shape=(output_length, output_dim))(
        fixed_r_decoder_input)


    encoder = Bidirectional(GRU(hidden_dim), merge_mode="ave", name="bidirectional1")
    q_encoder_output = encoder(q_encoder_input_masked)
    q_encoder_output = Dropout(rate=drop_rate, name="dropout1")(q_encoder_output)

    decoder = Bidirectional(GRU(hidden_dim), merge_mode="ave", name="bidirectional2")
    r_decoder_output = decoder(fixed_r_decoder_input_masked)
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


def read_questions(qa_file):
    question_answers = []
    with open(qa_file, "r") as f:
        ii = 0
        for line in f:
            l = line.strip()
            if l != "":
                if ii % 2 == 0:
                    question_text = l
                    qid = "question-%d" % (ii/2)
                else:
                    did_arr = l.split(" ")
                    answer_ids = []
                    for did in did_arr:
                        if did!= "": # twitter and ebay has double spaces
                            answer_ids.append("doc-%s" % did)

                    question = Question(qid, question_text, answer_ids)
                    question_answers.append(question)
            ii += 1
    return question_answers

def read_docs(doc_file):
    documents = []
    with open(doc_file, "r") as f:
        ii = 1 # doc id starts from 1
        for line in f:
            l = line.strip()
            if l!="":
                doc_id = "doc-%d" % ii
                doc = Document(doc_id, l)
                documents.append(doc)
            ii += 1
    return documents

def get_stemmed_words(sentence):
    sentence = "".join(filter(lambda x: x in printable, sentence.lower()))
    # sentence = sentence.encode('ascii', errors='ignore')

    sentence_words = WordPunctTokenizer().tokenize(sentence)
    sentence_words = [SnowballStemmer('english').stem(word) for word in sentence_words]
    return sentence_words


def train_w2v(questions, documents, to_w2v_path):
    if not os.path.exists(to_w2v_path):
        corpus = []
        workers = multiprocessing.cpu_count()

        # https://www.journaldev.com/19279/python-gensim-word2vec
        for q in questions:
            sentence_words = get_stemmed_words(q.question_text)
            corpus.append(sentence_words)
        for d in documents:
            sentence_words = get_stemmed_words(d.doc_text)
            corpus.append(sentence_words)

        # the parameters are same with BIKER in the paper
        model = Word2Vec(corpus, min_count=5, alpha=0.025, batch_words=10000, cbow_mean=1, compute_loss=False,
                         min_alpha=0.0001, negative=5, sg=0, window=5, workers=workers, size=100)
        model.save(to_w2v_path)
        print("saved to ", to_w2v_path)

    model = Word2Vec.load(to_w2v_path)
    return model


def init_doc_matrix(doc,w2v, length):
    matrix = np.zeros((length,100)) #word embedding size is 100
    for i, ww in enumerate(doc):
        if i >= length:
            break
        if ww in w2v.wv.vocab:
            matrix[i] = np.array(w2v.wv[ww])

    #l2 normalize
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(length, 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
        #matrix = matrix / np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
    except RuntimeWarning:
        print(doc)

    #matrix = np.array(preprocessing.normalize(matrix, norm='l2'))
    return matrix

# get the matrix
def preprocess_all_questions(questions, w2v):
    max_len = 0
    for question in questions:
        title_words = get_stemmed_words(question.question_text)
        if title_words[-1] == '?':
            title_words = title_words[:-1]
        max_len = max(max_len, len(title_words))

    processed_questions = list()
    for question in questions:
        title_words = get_stemmed_words(question.question_text)
        if title_words[-1] == '?':
            title_words = title_words[:-1]

        question.words = title_words
        question.matrix = init_doc_matrix(question.words, w2v, max_len)
        processed_questions.append(question)
    return processed_questions

# get the matrix
def preprocess_all_documents(questions, documents, w2v):
    doc_cnt = {}
    for q in questions:
        for did in q.answer_ids:
            if did in doc_cnt.keys():
                doc_cnt[did] += 1
            else:
                doc_cnt[did] = 1

    doc_weight = {}
    t_max = 0
    t_min = -1
    for k in doc_cnt.keys():
        t_max = max(t_max, doc_cnt[k])
        if t_min == -1:
            t_min = doc_cnt[k]
        else:
            t_min = min(t_min, doc_cnt[k])
    for k in doc_cnt.keys():
        doc_weight[k] = 1.0 * doc_cnt[k] / t_max
    w_min = 1.0 * t_min / t_max
    print("=== t_min:%d, t_max:%d, w_min:%.4f" % (t_min, t_max, w_min))
    logging.info("=== t_min:%d, t_max:%d, w_min:%.4f" % (t_min, t_max, w_min))

    max_length = 0
    for doc in documents:
        words = get_stemmed_words(doc.doc_text)
        if words[-1] == '?':
            words = words[:-1]
        max_length = max(max_length, len(words))

    processed_ducuments = list()
    for doc in documents:
        words = get_stemmed_words(doc.doc_text)
        if words[-1] == '?':
            words = words[:-1]

        doc.words = words
        doc.matrix = init_doc_matrix(doc.words, w2v, max_length)
        if doc.id in doc_cnt.keys():
            doc.count = doc_cnt[doc.id]
        if doc.id in doc_weight.keys():
            doc.weight = doc_weight[doc.id]
        else:
            doc.weight = w_min
        processed_ducuments.append(doc)
    return processed_ducuments

def train_nn(questions, documents, args, train_num):
    logger.info("=== preprocessing...")
    ns_amount = args.ns_amount

    documents_dict = {}

    # 选取数据集里出现过的 documents，用作训练。因为大部分 API 几乎很少被用到。
    documents_selected = []

    for doc in documents:
        doc_id = doc.id
        documents_dict[doc_id] = doc
        if doc.count > 0:
            # print("%s : %d" % (doc.id, len(doc.words)))
            documents_selected.append(doc)



    input_length = questions[0].matrix.shape[0]
    output_length = documents[0].matrix.shape[0]
    print("=== input_length: %d, output_length: %d" % (input_length, output_length) )
    logger.info("=== input_length: %d, output_length: %d" % (input_length, output_length) )

    questions_train = questions[:train_num]
    questions_test = questions[train_num:]

    batch_size = args.batch_size
    def generator():
        # Create empty arrays to contain batch of features and labels#

        # [q_encoder_input, r_decoder_input, w_decoder_input, weight_data_r, weight_data_w]

        while True:
            q_encoder_input = []
            r_decoder_input = []
            w_decoder_input = []
            weight_data_r = []
            weight_data_w = []
            y_data = []

            for i in range(batch_size):
                q = random.choice(questions_train)

                y = [1] + [0] * ns_amount
                y_data.append(y)

                q_encoder_input.append(q.matrix)

                # 每个question一个正确答案
                aid = q.answer_ids[0]
                r_decoder_input.append(documents_dict[aid].matrix)
                weight_data_r.append(documents_dict[aid].weight)

                # 10个un-related答案
                u_aids = []
                if args.doc_mode == 0: # for all
                    for i in range(ns_amount):
                        r = random.randint(1, len(documents) - 1)
                        while (documents[r].id in q.answer_ids):
                            r = random.randint(1, len(documents) - 1)
                        u_aids.append(documents[r].id)
                else:
                    for i in range(ns_amount): # for those appear on the SO
                        r = random.randint(1, len(documents_selected) - 1)
                        while (documents_selected[r].id in q.answer_ids):
                            r = random.randint(1, len(documents_selected) - 1)
                        u_aids.append(documents_selected[r].id)

                w_decoder = []
                w_weight = []
                for id in u_aids:
                    w_decoder.append(documents_dict[id].matrix)
                    w_weight.append(documents_dict[id].weight)

                w_decoder = np.array(w_decoder).reshape(output_length, args.input_dim, ns_amount)
                w_weight = np.array(w_weight).reshape((1, ns_amount))
                w_decoder_input.append(w_decoder)
                weight_data_w.append(w_weight)

            y_data = np.array(y_data).reshape(batch_size, (1 + ns_amount))
            q_encoder_input = np.array(q_encoder_input)
            r_decoder_input = np.array(r_decoder_input)
            w_decoder_input = np.array(w_decoder_input)
            weight_data_r = np.array(weight_data_r)
            weight_data_w = np.array(weight_data_w)
            logging.info("=== one batch generated")

            yield [q_encoder_input, r_decoder_input, w_decoder_input, weight_data_r, weight_data_w], y_data


    print("start training...")
    logger.info("=== start training...")

    model = negative_samples(input_length=input_length,
                             input_dim=args.input_dim,
                             output_length=output_length,
                             output_dim=args.output_dim,
                             hidden_dim=args.hidden_dim,
                             ns_amount=ns_amount,
                             learning_rate=args.learning_rate,
                             drop_rate=args.drop_rate, )
    print(model.summary())

    checkpoint = ModelCheckpoint("ckpt/best_model.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=10)

    model.fit_generator(generator(), epochs=args.epochs, steps_per_epoch= train_num // batch_size, verbose=1, callbacks=[checkpoint])

    # for test
    q_encoder_input = []
    r_decoder_input = []
    w_decoder_input = []
    weight_data_r = []
    weight_data_w = []
    y_data = []
    for q in questions_test:
        y = [1] + [0] * ns_amount
        y_data.append(y)

        q_encoder_input.append(q.matrix)

        # 每个question一个正确答案
        aid = q.answer_ids[0]
        r_decoder_input.append(documents_dict[aid].matrix)
        weight_data_r.append(documents_dict[aid].weight)

        # 10个un-related答案
        u_aids = []
        for i in range(10):
            r = random.randint(1, len(documents) - 1)
            while (documents[r].id in q.answer_ids):
                r = random.randint(1, len(documents) - 1)
            u_aids.append(documents[r].id)

        w_decoder = []
        w_weight = []
        for id in u_aids:
            w_decoder.append(documents_dict[id].matrix)
            w_weight.append(documents_dict[id].weight)

        w_decoder = np.array(w_decoder).reshape(output_length, args.input_dim, ns_amount)
        w_weight = np.array(w_weight).reshape((1, ns_amount))
        w_decoder_input.append(w_decoder)
        weight_data_w.append(w_weight)

    y_data = np.array(y_data).reshape(len(questions_test), (1 + ns_amount))

    res = model.evaluate([q_encoder_input, r_decoder_input, w_decoder_input,
                          weight_data_r, weight_data_w], y_data, verbose=1)
    print("=== training over.")
    logger.info("training over")
    print(model.metrics_names)
    print(res)
    print(model.summary())

    model.save(args.model_path)
    print("saved model to:", args.model_path)
    logging.info("=== saved model to: %s" % args.model_path)

    model.save_weights(args.weight_path)
    print("saved weights to: %s" % args.weight_path)
    logging.info("=== saved weights to: %s" % args.weight_path)


# top 50
def test_nn(questions, documents, args, train_num, model_path, to_pred_file):
    logger.info("=== preprocessing...")
    ns_amount = args.ns_amount

    documents_dict = {}

    # 选取数据集里出现过的 documents，用作训练。因为大部分 API 几乎很少被用到。
    documents_selected = []

    for doc in documents:
        doc_id = doc.id
        documents_dict[doc_id] = doc
        if doc.count > 0:
            # print("%s : %d" % (doc.id, len(doc.words)))
            documents_selected.append(doc)

    input_length = questions[0].matrix.shape[0]
    output_length = documents[0].matrix.shape[0]
    print("=== input_length: %d, output_length: %d" % (input_length, output_length))
    logger.info("=== input_length: %d, output_length: %d" % (input_length, output_length))


    questions_test = questions[train_num:]


    model = negative_samples(input_length=input_length,
                             input_dim=args.input_dim,
                             output_length=output_length,
                             output_dim=args.output_dim,
                             hidden_dim=args.hidden_dim,
                             ns_amount=ns_amount,
                             learning_rate=args.learning_rate,
                             drop_rate=args.drop_rate, )
    model.load_weights(model_path)

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


    # for test
    for q in questions_test:
        print("now: %s" % q.id)
        q_encoder_input = []
        r_decoder_input = []
        weight_data_r = []

        for doc in documents_selected:
            q_encoder_input.append(q.matrix)
            r_decoder_input.append(doc.matrix)
            weight_data_r.append(doc.weight)

        res = new_nn_model.predict([q_encoder_input, r_decoder_input, weight_data_r])


        with open(to_pred_file, "a") as fw:
            for i, row in enumerate(res):
                api = documents_selected[i].id
                score = row[0]
                fw.write("%s\tQ0\t%s\t%d\t%.8f\t%s\n" % (q.id, api, i + 1, score, "indri"))
        print("saved to", to_pred_file)

def generate_ltr_training_data(questions, documents, args, train_num, to_file_path):
    logger.info("=== extract_features...")
    ns_amount = args.ns_amount

    documents_dict = {}

    # 选取数据集里出现过的 documents，用作训练。因为大部分 API 几乎很少被用到。
    documents_selected = []

    for doc in documents:
        doc_id = doc.id
        documents_dict[doc_id] = doc
        if doc.count > 0:
            # print("%s : %d" % (doc.id, len(doc.words)))
            documents_selected.append(doc)

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
        # print("layer %d: " % i, weights)

    # 这个 model 用于提取 QA representations
    qa_vec_model = Model(inputs=new_nn_model.input, outputs=new_nn_model.get_layer('output_hid').output)

    step = 0
    while step * 200 <= train_num:
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
            question = questions[ss]
            logger.info("now: %s" % question.id  )
            qid_list.append(question.id)
            label_list.append(1)

            y = 1
            y_data.append(y)

            q_encoder_input.append(question.matrix)

            # 每个question一个正确答案
            aid = question.answer_ids[0]
            aid_list.append(aid)
            r_decoder_input.append(documents_dict[aid].matrix)
            weight_data_r.append(documents_dict[aid].weight)

            # 10个un-related答案
            u_aids = []
            for i in range(10):
                r = random.randint(1, len(documents) - 1)
                while (documents[r].id in q.answer_ids):
                    r = random.randint(1, len(documents) - 1)
                u_aids.append(documents[r].id)

            for u_aid in u_aids:
                qid_list.append(question.id)
                label_list.append(0)
                aid_list.append(u_aid)

                doc = documents_dict[u_aid]

                # 这些答案都是unrelated
                y = 0
                y_data.append(y)

                q_encoder_input.append(question.matrix)
                r_decoder_input.append(doc.matrix)
                weight_data_r.append(doc.weight)

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

def get_nn_pred_top50(to_file_nn_pred_all, to_file_nn_pred, k=50):
    all_data = {}
    print("read: %s" % to_file_nn_pred_all)
    with open(to_file_nn_pred_all, "r") as f:
        for line in f:
            l = line.strip()
            if l!="":
                qid, Q0, api, rank, score, indri = l.split("\t")
                score = float(score)
                if qid in all_data.keys():
                    if api not in all_data[qid]:
                        all_data[qid].append([api, score])
                else:
                    all_data[qid] = [ [api, score] ]
    with open(to_file_nn_pred, "w") as fw:
        for qid in all_data.keys():
            rank_list = all_data[qid]
            rank_list = sorted(rank_list, key=lambda item: item[1], reverse=True)
            ii = 1
            for api, score in rank_list[:k]:
                fw.write("%s\tQ0\t%s\t%d\t%.8f\t%s\n" % (qid, api, ii, score, "indri"))
                ii += 1
        print("saved to %s" % to_file_nn_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--data_type', help='data_type', type=str, default='stackoverflow3')

    parser.add_argument('--input_dim', help='input_dim', type=int, default=100)
    parser.add_argument('--output_dim', help='output_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', help='hidden_dim', type=int, default=64)
    parser.add_argument('--ns_amount', help='ns_amount', type=int, default=10)
    parser.add_argument('--doc_mode', help='doc_mode', type=int, default=0) # 0:for all,  1:for those apprear on the SO

    parser.add_argument("--pool_s", default=20, type=int, help="")
    parser.add_argument("--pool_stride", default=5, type=int, help="")

    parser.add_argument('--learning_rate', help='learning_rate', type=float, default=0.001)
    parser.add_argument('--drop_rate', help='drop_rate', type=float, default=0.01)

    parser.add_argument('--batch_size', help='batch_size', type=int, default=32)
    parser.add_argument('--epochs', help='epochs', type=int, default=100)

    parser.add_argument('--qa_file', help='qa_file', type=str, default='stackoverflow3/QA_list.txt')
    parser.add_argument('--doc_file', help='doc_file', type=str, default='stackoverflow3/Doc_list.txt')
    parser.add_argument('--model_path', help='model_path', type=str, default='models/nn_model_QA2021_stackoverflow3.bin')
    parser.add_argument('--weight_path', help='weight_path', type=str, default='ckpt/nn_weights_QA2021_stackoverflow3.h5')

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

    w2v_path = "models/w2v_%s.bin" % data_type

    to_file_nn_pred_all = "data/nn_%s_pred_all.txt" % data_type
    to_file_nn_pred = "data/nn_%s_pred_top50.txt" % data_type

    to_file_pred = "data/QA2021_%s_pred.txt" % data_type
    to_file_qrel = "data/QA2021_%s_qrel.txt" % data_type

    question_answers = read_questions(args.qa_file)
    documents = read_docs(args.doc_file)

    train_num = int(len(question_answers) * 0.9)
    print("total: %d, train: %d" % (len(question_answers), train_num))

    w2v = train_w2v(question_answers[:train_num], documents, w2v_path)

    questions = preprocess_all_questions(question_answers, w2v)
    documents = preprocess_all_documents(question_answers, documents, w2v)

    # stackoverflow statistics
    # print("files: %s" % args.qa_file)
    # all_words = []
    # que_length_total = 0
    # doc_length_total = 0
    # max_q_length = 0
    # max_d_length = 0
    # for q in questions:
    #     que_length_total += len(q.words)
    #     max_q_length = max(max_q_length, len(q.words))
    #     all_words = all_words + q.words
    # for d in documents:
    #     doc_length_total += len(d.words)
    #     max_d_length = max(max_d_length, len(d.words))
    #     all_words = all_words + d.words
    # all_words_unique = set(all_words)
    # print("total words: %d" % len(all_words))
    # print("total unique words: %d" % len(all_words_unique))
    # print("number of questions: %d" % len(questions))
    # print("max length of questions: %d" % max_q_length)
    # print("average length of questions: %.2f" % (1.0*que_length_total/len(questions)))
    # print("number of documents: %d" % len(documents))
    # print("max length of documents: %d" % max_d_length)
    # print("average length of documents: %.2f" % (1.0*doc_length_total / len(documents)))

    # cnt_of_0 = 0
    # for doc in documents:
    #     if doc.count == 0:
    #         print(doc.id)
    #         cnt_of_0 += 1
    # print("cnt_of_0: %d" % cnt_of_0)


    # nn_model = train_nn(questions, documents, args, train_num)



    # Evaluation:
    if not os.path.exists(to_file_qrel):
        with open(to_file_qrel, "w") as fw:
            for q in questions[train_num:]:
                for doc_id in set(q.answer_ids):
                    fw.write("%s 0 %s 1\n" % (q.id, doc_id))
        print("saved to %s" % to_file_qrel)

    test_nn(questions, documents, args, train_num, "ckpt/best_model_epoch10_negative10.hdf5", to_file_nn_pred_all)

    if not os.path.exists(to_file_nn_pred):
        get_nn_pred_top50(to_file_nn_pred_all, to_file_nn_pred)

