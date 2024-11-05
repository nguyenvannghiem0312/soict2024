import logging
import math
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample
from utils.io import read_json_or_dataset
from utils.dataset import search_by_id

import argparse
import json

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

# Load your custom training data
logger.info("Loading custom training data")

def load_custom_data(config):
    filepath = config['train_path']
    train_samples = []

    data = read_json_or_dataset(filepath)
    for item in data:
        query = item['text']
        for relevant in item['relevant']:
            train_samples.append(InputExample(texts=[query, relevant['text']], label=1))
        for relevant in item['not_relevant']:
            train_samples.append(InputExample(texts=[query, relevant['text']], label=0))

    return train_samples

def load_eval(config):
    corpus_dev = read_json_or_dataset(config["corpus_dev_path"])
    query_dev = read_json_or_dataset(config["query_dev_path"])
    cross_dev = read_json_or_dataset(config["cross_dev_path"])

    dev_samples = []

    for item in cross_dev:
        query = read_json_or_dataset(data=query_dev, search_id=item['id'])['text']
        relevants = read_json_or_dataset(data=query_dev, search_id=item['id'])['relevant']

        positive = [rel['text'] for rel in relevants]
        negative = []
        for id in item['relevant']:
            context = read_json_or_dataset(data=corpus_dev, search_id=id)['text']
            if context not in positive:
                negative.append(context)

        dev_samples.append({
            'query': query,
            'positive': positive,
            'negative': negative
        })
    
    return dev_samples


def train(config):
    train_samples, dev_samples = load_custom_data(config)
    train_batch_size = config['batch_size']
    num_epochs = config['num_train_epochs']
    model_save_path = config['output_dir'] + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = CrossEncoder(config['model'], trust_remote_code=True)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    
    if dev_samples:
        dev_evaluator = CERerankingEvaluator(samples=dev_samples, mrr_at_k=10)
    else:
        evaluator = None

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * config['warmup_ratio'])
    logger.info(f"Warmup-steps: {warmup_steps}")

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=config['eval_steps'] if evaluator else None,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        evaluation_steps=config['eval_steps'],
        save_best_model=config['load_best_model_at_end'],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training cross with configuration.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/configs/cross.json", 
        help="Path to the configuration JSON file."
    )
    
    args = parser.parse_args()
    config = read_json_or_dataset(args.config_path)

    print("Config: ", json.dumps(config, indent=4, ensure_ascii=False))

    train(config=config)