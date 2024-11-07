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

def threshold_filter(scores, threshold, ratio):
    return (scores[0] - scores[1] >= threshold and 
            (scores[0] - scores[1]) / (scores[1] - scores[2] + 0.00001) >= ratio)
    
def load_eval(config, threshold=0.2, ratio=2):
    corpus_dev = read_json_or_dataset(config["corpus_dev_path"])
    query_dev = read_json_or_dataset(config["query_dev_path"])
    cross_dev = read_json_or_dataset(config["cross_dev_path"])

    dev_samples = []
    no_rerank_mrr = []  # no reranking

    for item in cross_dev:
        try:
            query = search_by_id(data=query_dev, search_id=item['id'])['text']
            relevants = search_by_id(data=query_dev, search_id=item['id'])['relevant']
            scores = item['score']

            positive = [rel['text'] for rel in relevants]
            negative = []
            for id in item['relevant']:
                context = search_by_id(data=corpus_dev, search_id=id)['text']
                if context not in positive:
                    negative.append(context)

            if threshold_filter(scores, threshold, ratio) == False:
                dev_samples.append({
                    'query': query,
                    'positive': positive,
                    'negative': negative,
                })
            else:
                positive_ids = [rel['id'] for rel in relevants]
                
                relevant_predict = item['relevant']
                
                for idx, id_pred in enumerate(relevant_predict[:10]):
                    if id_pred in positive_ids:
                        no_rerank_mrr.append(1 / (1 + idx))
                        break
        except:
            continue
    logger.info(f"MRR score no rerank: {sum(no_rerank_mrr) / (len(cross_dev) - len(dev_samples))}")
    return dev_samples, no_rerank_mrr, len(cross_dev)


def train(config):
    model_save_path = config['output_dir'] + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = CrossEncoder(config['model'], trust_remote_code=True)
    
    dev_samples, no_rerank_mrr, len_samples = load_eval(config=config)
    logger.info(f"DEV: {len(dev_samples)}")
    logger.info(f"No rerank: {len(no_rerank_mrr)}")
    if dev_samples:
        dev_evaluator = CERerankingEvaluator(samples=dev_samples, mrr_at_k=10)
    else:
        dev_evaluator = None

    if config['only_test'] == True:
        mrr = dev_evaluator(model = model)
        mrr_scores = (mrr * len(dev_samples) + sum(no_rerank_mrr)) / len_samples
        print(mrr_scores)
        return None
    
    train_samples = load_custom_data(config)
    train_batch_size = config['batch_size']
    num_epochs = config['num_train_epochs']

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * config['warmup_ratio'])
    logger.info(f"Warmup-steps: {warmup_steps}")

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=config['eval_steps'] if dev_evaluator else None,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
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