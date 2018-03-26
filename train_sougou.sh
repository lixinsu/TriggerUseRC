#!/bin/bash
set -ex 


# format data to pqa 
TASK=$1
#MODEL=use_sample_data
MODEL=sougou_data

#for d in train dev
#do
#    python3 scripts/convert/format_DR_chn.py data/${TASK}/${d}.json data/${TASK}/DR_${d}.json
#done
#
#for d in train dev
#do
#    python3 scripts/reader/preprocess.py  data/${TASK} data/${TASK} --split DR_${d} --workers 20 --tokenizer jieba --standard 1
#done
#
#
#
#exit 
python3 scripts/reader/train.py  --num-epochs 40 \
                    --model-dir data/models/${TASK} \
                    --model-name ${MODEL} \
                    --data-dir data/${TASK} \
                    --train-file DR_train-processed-jieba.txt \
                    --dev-file DR_dev-processed-jieba.txt \
                    --dev-json DR_dev.json \
                    --embed-dir data/embeddings \
                    --embedding-file sogou.full.embed \
                    --restrict-vocab 0 \
                    --standard 1
