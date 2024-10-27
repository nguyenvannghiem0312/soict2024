import csv
import gzip
import logging
import math
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator, CESoftmaxAccuracyEvaluator
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample
from utils.io import read_json

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

# Load your custom training data
logger.info("Loading custom training data")

def load_custom_data(config):
    filepath = config['train_path']
    train_samples = []
    dev_samples = [] 

    with open(filepath, 'r', encoding="utf8") as f:
        data = json.load(f)
        for item in data:
            query = item['text']
            for relevant in item['relevant']:
                train_samples.append(InputExample(texts=[query, relevant['text']], label=1))

    with open(config['query_dev_path'], 'r', encoding="utf8") as f:
        data = json.load(f)
        for item in data:
            query = item['text']
            dev_samples.append({'query': query, 'positive': [r['text'] for r in item['relevant']], 'negative': ['Tôi không biết']})

    return train_samples, dev_samples


def train(config):
    train_samples, dev_samples = load_custom_data(config)
    train_batch_size = config['batch_size']
    num_epochs = config['num_train_epochs']
    model_save_path = config['output_dir'] + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = CrossEncoder(config['model'], trust_remote_code=True)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    
    if dev_samples:
        evaluator = CERerankingEvaluator(dev_samples, name="custom-dev")
    else:
        evaluator = None

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * config['warmup_ratio'])
    logger.info(f"Warmup-steps: {warmup_steps}")

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=config['eval_steps'] if evaluator else None,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
    )

if __name__ == "__main__":
    config = read_json(path="configs/cross.json")
    train(config=config)