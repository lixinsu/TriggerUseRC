#!/bin/bash

TASK=$1
MODEL_PATH=$2

set -ex


python3 scripts/convert/format_DR.py data/${TASK}/test.json data/${TASK}/DR_test.json test

python3 scripts/reader/predict.py \
                        data/${TASK}/DR_test.json \
                        --model ${MODEL_PATH} \
                        --out-dir data/${TASK}/pred \
                        --standard  \
                        --batch-size 64 \
                        --tokenizer spacy

python3 scripts/reader/triggering.py 

