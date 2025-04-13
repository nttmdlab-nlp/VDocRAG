#!/bin/bash

QUERY_DATASET=dude
CORPUS_DATASET=dude
EMBEDDING_OUTPUT_DIR=outputs/vdocretriever-phi3-vision_finetune/embs

# encoding queries
CUDA_VISIBLE_DEVICES=0 python -m vdocrag.vdocretriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path microsoft/Phi-3-vision-128k-instruct \
  --lora_name_or_path NTT-hil-insight/VDocRetriever-Phi3-vision \
  --lora \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 24 \
  --query_max_len 256 \
  --dataset_name NTT-hil-insight/OpenDocVQA \
  --dataset_config $QUERY_DATASET \
  --dataset_split test \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/query-${QUERY_DATASET}.pkl

# encoding documents
for s in 0 1 2 3
do
CUDA_VISIBLE_DEVICES=0 python -m vdocrag.vdocretriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path microsoft/Phi-3-vision-128k-instruct \
  --lora_name_or_path NTT-hil-insight/VDocRetriever-Phi3-vision \
  --lora \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 4 \
  --corpus_name NTT-hil-insight/OpenDocVQA-Corpus \
  --corpus_config $CORPUS_DATASET \
  --corpus_split test \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/corpus.${CORPUS_DATASET}.${s}.pkl 
done

# retrieving documentss
python -m vdocrag.vdocretriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/query-${QUERY_DATASET}.pkl \
    --document_reps $EMBEDDING_OUTPUT_DIR/corpus.${CORPUS_DATASET}'.*.pkl' \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/rank.${QUERY_DATASET}.${CORPUS_DATASET}.txt \