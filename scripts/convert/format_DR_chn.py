#!/usr/bin/env python
# coding: utf-8

import json
import sys



def strQ2B(ustring):
    """
    python3 全角转半角
   """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def format_pqa(infile, outfile, mode='train'):
    print(outfile)
    fo = open(outfile, 'w')
    #debug = 1000
    cnt_p, total = 0.0, 0.0
    with open(infile) as infp:
        for line in infp:
            #if debug == 0:
            #    break
            #debug -= 1
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            a = strQ2B(data['answer'].lower())
            q = strQ2B(data['query'].lower())
            qid = data['query_id']
            if len(a.strip()) == 0 or len(q)==0:
                print('error data format')
                continue
            for i, p in enumerate(data['passages']):
                p_text = strQ2B(p['passage_text'])
                p_id = p['passage_id']
                label = 1 if a in p_text else 0
                if label == 1:
                    cnt_p += 1
                total +=1
                if mode == 'train':
                    if len(p_text) > 0:
                        fo.write(json.dumps({'qid': "%s-%s" % (qid, p_id), 'passage': p_text, 'query': q, 'answers':[{'text':a, 'answer_start': -1}], 'label': label}, ensure_ascii=False) + '\n')
                elif mode == 'test':
                    if len(p_text) > 0:
                        fo.write(json.dumps({'qid': "%s-%s" % (qid, p_id), 'passage': p_text, 'query': q, 'answers':[{'text':a, 'answer_start': -1}], 'label': label}, ensure_acii=False) + '\n')
    print('pos label rate: %.3f ' % (cnt_p / total))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        format_pqa(sys.argv[1], sys.argv[2])
    elif len(sys.argv)==4:
        format_pqa(sys.argv[1], sys.argv[2], mode=sys.argv[3])
