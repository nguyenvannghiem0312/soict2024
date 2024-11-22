cd /workspace/src

echo ">>> Infer e5 base trim vocab v3 on public test"
sed -i 's|Turbo-AI/me5-base-v3__trim-vocab|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v2|g' configs/infer_sbert.json
sed -i 's|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v2|Turbo-AI/multilingual-e5-base-trimm-vocab-1024-v3|g' configs/infer_sbert.json
sed -i 's|Turbo-AI/data-train|Turbo-AI/data-public_test|g' configs/infer_sbert.json
python semantic_search/inference_sbert.py --config_path "configs/infer_sbert.json"
