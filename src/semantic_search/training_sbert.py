import logging
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import SimilarityFunction, InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers

from utils.dataset import process_data, process_dev
from utils.io import read_json_or_dataset, save_to_json

import argparse
import json


logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def load_config(config_path="configs/sbert.json"):
    """Load the configuration from a JSON file."""
    config = read_json_or_dataset(config_path)
    return config

def load_model(model_name):
    """Load the SentenceTransformer model."""
    model = SentenceTransformer(model_name)
    logging.info(model)
    return model


def load_datasets(train_path):
    """Load the NLI training and evaluation datasets."""
    train_datasets = read_json_or_dataset(train_path)
    train_datasets = process_data(train=train_datasets)
    
    datasets = {}
    anchor = [item["anchor"] for item in train_datasets]
    positive = [item["positive"] for item in train_datasets]

    datasets["anchor"] = anchor
    datasets["positive"] = positive
    if "negative" in train_datasets[0]:
        negative = [item["negative"] for item in train_datasets]
        datasets["negative"] = negative
    
    return Dataset.from_dict(datasets)


def define_loss(model, config, guide=None):
    """Define the Loss."""
    loss_type = config["loss"]

    if loss_type == "MultipleNegativesRankingLoss":
        return losses.MultipleNegativesRankingLoss(model)

    elif loss_type == "CachedMultipleNegativesRankingLoss":
        return losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=config["mini_batch_size"])

    elif loss_type == "CachedMultipleNegativesSymmetricRankingLoss":
        return losses.CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=config["mini_batch_size"])

    elif loss_type == "GISTEmbedLoss":
        return losses.GISTEmbedLoss(model=model, guide=guide)

    elif loss_type == "CachedGISTEmbedLoss":
        return losses.CachedGISTEmbedLoss(model=model, guide=guide, mini_batch_size=config["mini_batch_size"])

    else:
        raise ValueError(f"Loss type {loss_type} is not recognized.")



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


def define_training_args(config):
    """Define the training arguments."""
    return SentenceTransformerTrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        warmup_ratio=config["warmup_ratio"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        logging_steps=config["logging_steps"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        learning_rate=config["learning_rate"],
        run_name=config["run_name"],
    )


def train_model(model, args, train_dataset, train_loss, dev_evaluator):
    """Train the model using SentenceTransformerTrainer."""
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
    )
    trainer.train()


def main(config):
    model = load_model(config["model"])
    guide = None
    if config["guide_model"] != "":
        guide = load_model(config["guide_model"])

    train_dataset= load_datasets(train_path=config["train_path"])
    train_loss = define_loss(model=model, config=config, guide=guide)
    dev_evaluator = load_evaluator(corpus_dev_path=config["corpus_dev_path"], 
                                   query_dev_path=config["query_dev_path"])
    args = define_training_args(config=config)

    # Start training the model
    train_model(model, args, train_dataset, train_loss, dev_evaluator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training SBERT with configuration.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/configs/sbert.json", 
        help="Path to the configuration JSON file."
    )
    
    args = parser.parse_args()
    config = read_json_or_dataset(args.config_path)

    print("Config: ", json.dumps(config, indent=4, ensure_ascii=False))

    main(config)