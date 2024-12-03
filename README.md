# Soict-2024

## Quick start

The pytorch version is 2.4.1, update NVIDIA driver for support CUDA 12.x or downgrade torch in docker but maybe not work correctly

```bash
# Build image and run the last model for public test to get the final result
docker build -t turbo-legal .
# For interactive run full pipeline from begin
docker run --rm -it --gpus all --net host turbo-legal /bin/bash
./run.sh
or ./run_train.sh to train model
```

## For run individual step
First, go into `src` folder
```
cd src
```

Then change the model name, batch size, top k, epoch,... in config files (for more detail read the report) and run:

  - run trimm vocab
```
python vocab_trimm.py --config_path "configs/trimm_vocab_config.json"
```

  - run bm25
```
python fulltext_search/inference_bm25.py --config_path "configs/bm25_config.json"
```

  - run bm25s
```
python fulltext_search/inference_bm25s.py --config_path "configs/bm25s_config.json"
```

  - run training sbert
```
python semantic_search/training_sbert.py --config_path "configs/sbert.json"
```

  - run inference sbert
```
python semantic_search/inference_sbert.py --config_path "configs/infer_sbert.json"
```

  - run eval sbert in dev dataset
```
python semantic_search/eval_sbert.py --config_path "configs/sbert.json"
```

  - run training cross
```
python cross_encoder/training_cross.py --config_path "configs/cross.json"
```

  - run inference cross
```
python cross_encoder/inference_cross.py --config_path "configs/infer_cross.json"
```

  - run inference combine score
```
python inferance_combine.py
```

  - run training doc2query
```
python doc2query/training_doc2query.py --config_path "configs/doc2query_config.json"
```

  - run inference doc2query
```
python doc2query/inference_doc2query.py --config_path "configs/infer_doc2query.json"
```
