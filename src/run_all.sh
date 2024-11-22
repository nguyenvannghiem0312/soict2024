#!/bin/bash

cd /workspace/src

echo ">>> Convert data to json format"
python read_dataset.py

echo ">>> Trim vocab e5 base"
python vocab_trimm.py

echo ">>> Train doc2query"
python doc2query/training_doc2query.py --config_path "configs/doc2query_config.json"

echo ">>> Infer doc2query to get dataset"
python doc2query/inference_doc2query.py --config_path "configs/infer_doc2query.json"

echo ">>> Trim vocab t5 doc2query"
sed -i 's|intfloat/multilingual-e5-base|doc2query/msmarco-vietnamese-mt5-base-v1|g' configs/trimm_vocab_config.json
sed -i 's|multilingual-e5-base__trimm-vocab|vi-t5-doc2query-trimm-vocab|g' configs/trimm_vocab_config.json
python vocab_trimm.py

echo ">>> Expand dim trim vocab e5 base"
python utils/extend_max_length.py

echo ">>> Train e5 base trim vocab 1024 to get v1"
python semantic_search/training_sbert.py --config_path "configs/sbert.json"

echo ">>> BM25s generate samples"
python fulltext_search/inference_bm25s.py --config_path "configs/bm25s_config.json"

echo ">>> Train e5 base trim vocab v1 to get v2"
sed -i 's|Turbo-AI/multilingual-e5-base__trim_vocab-1024|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v1|g' configs/sbert.json
python semantic_search/training_sbert.py --config_path "configs/sbert.json"

echo ">>> Generate samples use e5 base trim vocab v2"
sed -i 's|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v3|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v2|g' configs/infer_sbert.json
sed -i 's|Turbo-AI/me5-base-v3__trim-vocab|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v2|g' configs/infer_sbert.json
sed -i 's|"top_k": 10|"top_k": 5|g' configs/infer_sbert.json
python semantic_search/inference_sbert.py --config_path "configs/infer_sbert.json"

echo ">>> Train e5 base trim vocab v2 to get v3"
sed -i 's|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v1|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v2|g' configs/sbert.json
python semantic_search/training_sbert.py --config_path "configs/sbert.json"

echo ">>> Infer e5 base trim vocab v3 on public test"
sed -i 's|"top_k": 5|"top_k": 10|g' configs/infer_sbert.json
sed -i 's|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v2|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v3|g' configs/infer_sbert.json
sed -i 's|Turbo-AI/data-train|Turbo-AI/data-public_test|g' configs/infer_sbert.json
python semantic_search/inference_sbert.py --config_path "configs/infer_sbert.json"
