import json
from sentence_transformers import (
    SentenceTransformer,
)
from sentence_transformers.evaluation import SimilarityFunction, InformationRetrievalEvaluator
from utils.dataset import process_data, process_dev
from utils.io import read_json_or_dataset, save_to_json
from utils.model import load_model2vec

import argparse
import json

def load_config(config_path="configs/sbert.json"):
    """Load the configuration from a JSON file."""
    config = read_json_or_dataset(config_path)
    return config

def load_model(model_name):
    """Load the SentenceTransformer model."""
    if 'm2v' not in model_name:
        model = SentenceTransformer(model_name)
    else:
        model = load_model2vec(model_name)
    return model

def load_evaluator(corpus_dev_path, query_dev_path):
    """Load the benchmark dataset and create an evaluator."""
    corpus_dev = read_json_or_dataset(corpus_dev_path)
    query_dev = read_json_or_dataset(query_dev_path)

    dev_datasets = process_dev(corpus=corpus_dev, query=query_dev)
    dev_evaluator = InformationRetrievalEvaluator(
        queries=dev_datasets['queries'],
        corpus=dev_datasets['corpus'],
        relevant_docs=dev_datasets['relevant_docs'],
        main_score_function=SimilarityFunction.COSINE,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[10],
        precision_recall_at_k= [10],
        map_at_k= [10],
    )
    return dev_evaluator

def eval_model(model, dev_evaluator):
    """Eval the model using SentenceTransformerTrainer."""
    results = dev_evaluator(model)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run eval SBERT in Dev dataset with configuration.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/configs/sbert.json", 
        help="Path to the configuration JSON file."
    )
    
    args = parser.parse_args()
    config = read_json_or_dataset(args.config_path)

    print("Config: ", json.dumps(config, indent=4, ensure_ascii=False))

    model = load_model(config["model"])

    dev_evaluator = load_evaluator(corpus_dev_path=config["corpus_dev_path"], 
                                   query_dev_path=config["query_dev_path"])
    
    result = eval_model(model, dev_evaluator)
    print("Result: ", json.dumps(result, indent=4, ensure_ascii=False))