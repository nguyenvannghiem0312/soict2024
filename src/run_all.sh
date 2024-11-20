#!/bin/bash

cd /workspace/src

# Preprocess data


# Run BM25 inference
python fulltext_search/inference_bm25.py --config_path "configs/bm25_config.json"

# Run BM25s inference
python fulltext_search/inference_bm25s.py --config_path "configs/bm25s_config.json"

# Train SBERT model
python semantic_search/training_sbert.py --config_path "configs/sbert.json"

# Run SBERT inference
python semantic_search/inference_sbert.py --config_path "configs/infer_sbert.json"

# Evaluate SBERT on the development dataset
python semantic_search/eval_sbert.py --config_path "configs/sbert.json"

# Train Cross Encoder model
python cross_encoder/training_cross.py --config_path "configs/cross.json"

# Run Cross Encoder inference
python cross_encoder/inference_cross.py --config_path "configs/infer_cross.json"

# Train Doc2Query model
python doc2query/training_doc2query.py --config_path "configs/doc2query_config.json"

# Run Doc2Query inference
python doc2query/inference_doc2query.py --config_path "configs/infer_doc2query.json"
