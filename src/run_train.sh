#!/bin/bash

cd /workspace/src

echo ">>> Train Weakly-Supervised Pre-Training"
python semantic_search/training_sbert.py --config_path "configs/wealy_supervised_gte.json"

echo ">>> Train GTE"
python semantic_search/training_sbert.py --config_path "configs/train_gte.json"

echo ">>> Train Jina Reranker"
python cross_encoder/training_cross.py --config_path "configs/cross.json"