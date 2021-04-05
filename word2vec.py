#from gensim.models.keyedvectors import KeyedVectors
#
# model = KeyedVectors.load_word2vec_format('wordembedding/rob04.wv.cbow.d300.w10.n10.m10.i10.W.bin', binary=True)
#
# # 转换成 txt 格式
# model.save_word2vec_format('wordembedding/rob04.wv.cbow.d300.w10.n10.m10.i10.W.txt', binary=False)
#
# # 另存为 bin 文件
# model.save_word2vec_format('wordembedding/rob04.wv.cbow.d300.w10.n10.m10.i10.W22.bin', binary=True)

from gensim.models.keyedvectors import KeyedVectors
from nltk.tokenize import word_tokenize
import string
import os
from gensim.models import Word2Vec
import sys
import time
import logging
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
now_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=BASE_DIR + '/' + now_time + '.log')
logger = logging.getLogger(__name__)


def remove_punc(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def get_embeddings(data_type, qa_file, doc_file, args):
    corpus = []

    with open(qa_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line != "" and i % 2 == 0:
                line = remove_punc(line)
                words = word_tokenize(line)
                corpus.append(words)

    with open(doc_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().lower()
            if line != "":
                line = remove_punc(line)
                words = word_tokenize(line)
                corpus.append(words)


    data_type = data_type.lower()
    min_count = args.min_count
    size = args.input_dim
    window = args.window
    negative = args.negative
    sg = 0

    w2v_model = Word2Vec(corpus,
                         min_count=min_count,
                         size=size,
                         window=window,
                         negative=negative,
                         sg = sg)
    to_file = "models/%s.wv.cbow.d%d.w%d.n%d.bin" % (data_type, size, window,negative)
    w2v_model.wv.save_word2vec_format(to_file, binary=True)
    print("saved to:", to_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--data_type', help='data_type',  required=True)

    parser.add_argument('--input_dim', help='input_dim', type=int, default=200)
    parser.add_argument('--min_count', help='min_count', type=int, default=1)
    parser.add_argument('--window', help='window', type=int, default=10)
    parser.add_argument('--negative', help='negative', type=int, default=10)
    parser.add_argument('--sg', help='sg', type=int, default=0)

    parser.add_argument('--qa_file', help='qa_file', type=str, default='')
    parser.add_argument('--doc_file', help='doc_file', type=str, default='')

    args = parser.parse_args()
    logger.info("training parameters %s", args)

    data_type = args.data_type

    print("running word2vec.py, data_type: %s" % data_type)
    logger.info("running word2vec.py, data_type: %s" % data_type)


    # doc_file = data_type + "/Doc_list.txt"
    # qa_file = data_type + "/QA_list.txt"
    get_embeddings(data_type, args.qa_file, args.doc_file, args)



