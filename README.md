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

run training cross
```
python cross_encoder/training_cross.py --config_path "configs/cross.json"
```

run inference cross
```
python cross_encoder/inference_cross.py --config_path "configs/infer_cross.json"
```

run training doc2query
```
python doc2query/training_doc2query.py --config_path "configs/doc2query_config.json"
```

run inference doc2query
```
python doc2query/inference_doc2query.py --config_path "configs/infer_doc2query.json"
```