#!/bin/bash
set -ex 


# format data to pqa 
TASK=$1
#MODEL=use_sample_data
MODEL=balance_sample_data

for d in train val
do
    python3 scripts/convert/format_DR.py data/${TASK}/sample_${d}.json data/${TASK}/DR_${d}.json
done

for d in train val
do
    echo "process ${d}"
    python3 scripts/reader/preprocess.py  data/${TASK} data/${TASK} --split DR_${d} --workers 5 --tokenizer spacy --standard 1
done

exit 


python3 scripts/reader/train.py  --num-epochs 40 \
                    --model-dir data/models/${TASK} \
                    --model-name ${MODEL} \
                    --data-dir data/${TASK} \
                    --train-file DR_train-processed-spacy.txt \
                    --dev-file DR_val-processed-spacy.txt \
                    --dev-json DR_val.json \
                    --embed-dir data/embeddings \
                    --standard 1
