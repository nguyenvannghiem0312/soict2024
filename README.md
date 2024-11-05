# Soict-2024

```
cd src
```

run bm25
```
python fulltext_search/inference_bm25.py --config_path "configs/bm25_config.json"
```

run bm25s
```
python fulltext_search/inference_bm25s.py --config_path "configs/bm25s_config.json"
```

run training sbert
```
python semantic_search/training_sbert.py --config_path "configs/sbert.json"
```

run inference sbert
```
python semantic_search/inference_sbert.py --config_path "configs/infer_sbert.json"
```

run eval sbert in dev dataset
```
python semantic_search/eval_sbert.py --config_path "configs/sbert.json"
```