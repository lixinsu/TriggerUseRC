#!/bin/bash

#python3 scripts/reader/triggering_eval.py data/searchqa/pred/DR_test-use_sample_data.preds data/searchqa/DR_test.json
#python3 scripts/reader/triggering_eval.py data/quasart/pred/DR_test-use_sample_data.preds data/quasart/DR_test.json
#
#python3 scripts/reader/triggering_eval.py data/quasart/pred/DR_test-balance_sample_data.preds data/quasart/DR_test.json
#python3 scripts/reader/triggering_eval.py data/searchqa/pred/DR_test-balance_sample_data.preds data/searchqa/DR_test.json

#python3 scripts/reader/triggering_eval.py data/sougou/pred/DR_test-sougou_data.preds data/sougou/DR_test.json
python3 scripts/reader/triggering_eval.py eval_result  --pred-file data/quasart/pred/DR_test-balance_sample_data.preds --DR-file data/quasart/DR_test.json --result-file data/quasart/merge_result.json
python3 scripts/reader/case_study.py merge_result --result-file data/quasart/merge_result.json --data_file data/quasart/sample_test.json --merge-file cast.txt
