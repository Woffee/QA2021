"""
https://github.com/usnistgov/trec_eval

http://www.rafaelglater.com/en/post/learn-how-to-use-trec_eval-to-evaluate-your-information-retrieval-system

@Time    : 2/1/21
@Author  : Wenbo
"""
import os
import io
import time
import logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import argparse

NUM = 30

def process_qrel(qrel_file, pred_file, args):
    qid_docids = {}

    with open(qrel_file, "r") as f:
        with open(to_file_qrel, "w") as fw:

            last_qid = 0
            data = []  # 记录一个question的记录

            for line in f.readlines():
                l = line.strip()
                if l != "":
                    arr = l.split(" ")
                    qid = int(arr[0])
                    did = arr[2]
                    rel = int(arr[3])

                    if last_qid == 0:
                        last_qid = qid

                    if qid == last_qid:
                        data.append([qid, did, rel])
                    else:
                        data = sorted(data, key=lambda k: k[2], reverse=True)
                        data = data[: NUM]
                        for row in data:
                            print(row)
                            # 记录每个 qid 下的 doc-id
                            if row[0] in qid_docids.keys():
                                qid_docids[row[0]].append(row[1])
                            else:
                                qid_docids[row[0]] = [row[1]]
                            fw.write("%d 0 %s %d\n" % (row[0], row[1], row[2]))
                        last_qid = qid
                        data = [[qid, did, rel]]

            data = sorted(data, key=lambda k: k[2], reverse=True)
            data = data[:20]
            for row in data:
                print(row)
                if row[0] in qid_docids.keys():
                    qid_docids[row[0]].append(row[1])
                else:
                    qid_docids[row[0]] = [row[1]]
                fw.write("%d 0 %s %d\n" % (row[0], row[1], row[2]))

    with open(pred_file, "r") as f:
        with open(to_file_pred, "w") as fw:
            last_qid = 0
            data = []  # 记录一个question的记录
            visited_did = []  # pred file 中，有的 doc id 会重复出现，这里去重

            for line in f.readlines():
                l = line.strip()
                if l != "":
                    arr = l.split(" ")
                    qid = int(arr[0])
                    did = arr[2]
                    score = float(arr[4])
                    cmt = arr[5]

                    if last_qid == 0:
                        last_qid = qid

                    if qid == last_qid:
                        if did not in visited_did:
                            data.append([qid, did, score, cmt])
                            visited_did.append(did)
                    else:
                        print("now: ", qid)
                        data = data[:200]
                        # data = sorted( data, key=lambda k: k[2], reverse=True )
                        r = 1
                        for row in data:
                            s_qid = row[0]
                            s_did = row[1]
                            if s_did in qid_docids[s_qid]:
                                print(row)
                                fw.write("%d\tQ0\t%s\t%d\t%.8f\t%s\n" % (row[0], row[1], r, row[2], row[3]))
                                r = r + 1
                        last_qid = qid
                        data = [[qid, did, score, cmt]]
                        visited_did = [did]

            r = 1
            for row in data:
                s_qid = row[0]
                s_did = row[1]
                if s_did in qid_docids[s_qid]:
                    print(row)
                    fw.write("%d\tQ0\t%s\t%d\t%.8f\t%s\n" % (row[0], row[1], r, row[2], row[3]))
                    r = r + 1
    print("done")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--qa_file', help='qa_file', type=str, default='/Users/woffee/www/emse-apiqa/our_code/stackoverflow3/QA_list.txt')
    parser.add_argument('--doc_file', help='doc_file', type=str, default='/Users/woffee/www/emse-apiqa/our_code/stackoverflow3/Doc_list.txt')

    parser.add_argument('--qrel_file', help='qrel_file', type=str, default='for_ltr/ltr_stackoverflow3_qrel.txt')
    parser.add_argument('--pred_file', help='pred_file', type=str, default='ltr/predicts/stackoverflow3_lambdaMART_MAP_pred.txt')
    parser.add_argument('--data_type', help='data_type', type=str, default='stackoverflow3')


    args = parser.parse_args()




    
    pred_file = args.pred_file

    to_file_qrel = "tmp/" + args.data_type + "-qrel-%d.txt" % NUM
    to_file_pred = "tmp/" + args.data_type + "-pred-%d.txt" % NUM


    if not os.path.exists("tmp"):
        os.mkdir("tmp")


    if not os.path.exists(to_file_pred):
        process_qrel(args.qrel_file, pred_file, args)


    trec_eval_path = '/Users/woffee/www/gis_technical_support/trec_eval-9.0.7/trec_eval'
    eva_cmd = "%s -m map -m P.1,3,5,10 -m recip_rank -m recall.1,3,5,10 %s %s" % ( trec_eval_path, to_file_qrel, to_file_pred)

    print(eva_cmd)
    print(" === %s - %d ===" % (args.data_type, NUM) )
    # os.system(eva_cmd)

    result = os.popen(eva_cmd)
    context = result.read()
    print(context)

    scores = []
    for line in context.splitlines():
        l = line.strip()
        arr = l.split("\t")
        scores.append(arr[-1])

    latex_str = "\\textcolor{blue}{%s}" % type
    for i in range(2, len(scores)):
        latex_str += " & \\textcolor{blue}{%s}" % scores[i]

    latex_str += " & \\textcolor{blue}{%s}" % scores[0]
    latex_str += " & \\textcolor{blue}{%s}" % scores[1]

    latex_str += " \\\\"
    print(latex_str)
    print("done")