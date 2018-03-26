#!/bin/bash

TASK=sougou
MODEL_PATH=$1

set -ex


python3 scripts/convert/format_DR.py data/${TASK}/gold_test.json data/${TASK}/DR_test.json test

python3 scripts/reader/predict.py \
                        data/${TASK}/DR_test.json \
                        --model ${MODEL_PATH} \
                        --out-dir data/${TASK}/pred \
                        --standard  \
                        --batch-size 32 \
                        --tokenizer jieba 

