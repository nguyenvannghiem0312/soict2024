cd /workspace/src

echo ">>> Infer gte base trim vocab v1 on private test"
python semantic_search/inference_sbert.py --config_path "configs/infer_sbert.json"

echo ">>> Infer jina reranker on private test"
python cross_encoder/inference_cross.py --config_path "configs/infer_cross.json"

echo ">>> Combine score gte and jina on private test"
python python inferance_combine.py
