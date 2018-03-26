#!/usr/bin/env python
# coding: utf-8

import sys
import json
from collections import defaultdict, Counter
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import fire
from prettytable import PrettyTable


def load_pred(pred_file):
    return json.loads(open(pred_file).read())


def load_origin(origin_file):
    data = []
    for line in open(origin_file):
        data.append(json.loads(line))
    return data


def calc_metrics(gt, pred):
    cmatrix = confusion_matrix(gt, pred)
    f1s  =  f1_score(gt, pred, average=None)
    recalls = recall_score(gt, pred, average=None)
    precisions = precision_score(gt, pred, average=None)
    return f1s, recalls, precisions, cmatrix



def auc_para(pred_scores, gts):
    """auc paragraph"""
    pred_scores = [y for x in pred_scores for y in x]
    gts = [y for x in gts for y in x]
    print("PARA: pos %s neg %s" % (sum(gts), len(gts)- sum(gts)))
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(gts, pred_scores)
    return metrics.auc(fpr, tpr), metrics.auc(recall, precision)


def auc_question(pred_scores, gts):
    """question level auc"""
    pred_scores =[max(x) for x in pred_scores]
    gts = [max(x) for x in gts]
    print("QUESTION: pos %s neg %s" % (sum(gts), len(gts)- sum(gts)))
    fpr, tpr, thresholds = metrics.roc_curve(gts, pred_scores, pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(gts, pred_scores)
    return metrics.auc(fpr, tpr), metrics.auc(recall, precision)


def eval_result(pred_file, DR_file, result_file):
    """merge predictions and origin data to result file"""
    preds = load_pred(pred_file)
    datas = load_origin(DR_file)

    qid2res = defaultdict(list)
    qid2gt = defaultdict(list)
    qids = []
    for data in datas:
        qpid = data['qid']
        qid =qpid.split('-')[0]
        if len(qids) == 0:
            qids.append(qid)
        elif qids[-1] != qid:
            qids.append(qid)
        qid2gt[qid].append(data['label'])
        qid2res[qid].append(preds[qpid][1])

    pred_scores, gts = [], []
    for qid in qids:
        pred_scores.append(qid2res[qid])
        gts.append(qid2gt[qid])

    p_roc, p_pr = auc_para(pred_scores, gts)
    q_roc, q_pr = auc_question(pred_scores, gts)
    table = PrettyTable(["p_roc", "p_pr", "q_roc", "q_pr"])
    table.add_row([p_roc, p_pr, q_roc, q_pr ])
    print(table)

    outf = open(result_file, 'w')
    for qid in qids:
        outf.write(json.dumps({'query_id':qid, 'predictions':qid2res[qid], 'ground_truth':qid2gt[qid]}) + '\n')




if __name__ == "__main__":
    fire.Fire()



