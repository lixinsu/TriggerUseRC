#!/usr/bin/env python
# coding: utf-8

import json
import sys
import pickle
from collections import defaultdict


def load_pred_file(pred_file):
    return json.loads(open(pred_file).read())


def load_origin_file(data_file):
    qids, examples = pickle.load(open(data_file, 'rb'))
    return qids, examples


def merge_answers(preds, qids, examples):
    qid2preds  = defaultdict(list)
    for qid in qids:
        print(qid.rsplit('-')[0])
        qid2preds[qid.rsplit('-')[0]].append((preds[qid]))
    return qid2preds


def generate_answer(qid2preds):
    qid2final = {}
    for qid, preds in qid2preds.items():
        finalpred = ''
        max_score = -10000
        for pred in preds[:50]:
            text,score = pred[0]
            if score > max_score:
                max_score = score
                finalpred = text
        qid2final[qid] = finalpred
    return qid2final


def save_to_disk(qid2final, filename):
    fo = open(filename, 'w')
    for qid, info in sorted(list(qid2final.items())):
        fo.write("%s\t%s\n" % (qid, info))

if __name__ == '__main__':
    preds = load_pred_file(sys.argv[1])
    qids, examples = load_origin_file(sys.argv[2])
    qid2preds = merge_answers(preds, qids, examples)
    qid2final = generate_answer(qid2preds)
    save_to_disk(qid2final, sys.argv[3])
