#!/usr/bin/env python
# coding: utf-8

import json
import sys

def format_pqa(infile, outfile, mode='train'):
    print(outfile)
    fo = open(outfile, 'w')
    with open(infile) as infp:
        for line in infp:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            a = data['answer'].lower()
            q = data['query'].lower()
            qid = data['query_id']
            if len(a.strip()) == 0 or len(q)==0:
                print('error data format')
                continue
            for i, p in enumerate(data['passages']):
                p_text = p['passage_text'].lower()
                #p_rank = p['rank']
                p_id = p['passage_id']
                if mode == 'train':
                    if a in p_text:
                        fo.write(json.dumps({'qid': "%s-%s" % (qid, p_id), 'passage': p_text, 'query': q, 'answers':[{'text':a, 'answer_start': p_text.find(a)}]}) + '\n')
                elif mode == 'test':
                    if len(p_text) > 0:
                        fo.write(json.dumps({'qid': "%s-%s" % (qid, p_id), 'passage': p_text, 'query': q, 'answers':[{'text':a, 'answer_start': -1}]}) + '\n')


if __name__ == "__main__":
    if len(sys.argv) == 3:
        format_pqa(sys.argv[1], sys.argv[2])
    elif len(sys.argv)==4:
        format_pqa(sys.argv[1], sys.argv[2], mode=sys.argv[3])
