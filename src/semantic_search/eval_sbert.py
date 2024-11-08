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

def load_model(config):
    """Load the SentenceTransformer model."""
    model = SentenceTransformer(config["model"], trust_remote_code=True)
    model.max_seq_length = config["max_length"]
    return model

def load_evaluator(config):
    """Load the benchmark dataset and create an evaluator."""
    corpus_dev = read_json_or_dataset(config["corpus_dev_path"])
    query_dev = read_json_or_dataset(config["query_dev_path"])

    dev_datasets = process_dev(corpus=corpus_dev, query=query_dev, query_prompt=config["query_prompt"], corpus_prompt=config["corpus_prompt"])
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
        show_progress_bar=True,
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

    model = load_model(config)

    dev_evaluator = load_evaluator(config)
    
    result = eval_model(model, dev_evaluator)
    print("Result: ", json.dumps(result, indent=4, ensure_ascii=False))