import logging
import warnings
import os
from tqdm.auto import tqdm
from utils.io import read_json_or_dataset, save_to_json

from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder

import argparse
import json
warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

def load_model(model_name='Alibaba-NLP/gte-multilingual-reranker-base', max_length=512):
    model = CrossEncoder(model_name=model_name, trust_remote_code=True, max_length=max_length)
    model.model.to('cuda')
    return model

from tqdm import tqdm
import os

def pipeline(model_name, corpus, predicts, output_k=10, max_length=512):
    def initialize_batch():
        return {
            "queries": [],
            "relevant_contexts": [],
            "predict_ids": [],
            "relevant_ids": [],
            "relevant_scores": []
        }

    def threshold_filter(scores, threshold, ratio):
        return (scores[0] - scores[1] >= threshold and 
                (scores[0] - scores[1]) / (scores[1] - scores[2] + 0.00001) >= ratio)

    def write_output(fo, predict_id, relevant, scores, enable_filter=False):
        sorted_results = sorted(zip(scores, relevant), key=lambda x: -x[0])[:output_k]
        if enable_filter and threshold_filter(scores, embed_threshold, embed_threshold_ratio):
            sorted_results = list(zip(scores[:output_k], relevant[:output_k]))
        
        fo.write(f"{predict_id} {' '.join([str(x[1]) for x in sorted_results])}\n")
        outputs.append({
            'id': predict_id,
            'text': public_test[predict_id],
            'relevant': [x[1] for x in sorted_results],
        })

    def process_batch(model, fo, batch, enable_filter=False):
        scores = model.predict(list(zip(batch['queries'], batch['relevant_contexts'])), batch_size=batch_size)
        assert len(scores) == len(batch['queries'])
        
        for i in range(len(batch['queries']) // topk):
            predict_id = batch['predict_ids'][i * topk]
            relevant = batch['relevant_ids'][i * topk:(i + 1) * topk]
            score_slice = scores[i * topk:(i + 1) * topk]
            write_output(fo, predict_id, relevant, score_slice, enable_filter=enable_filter)

    # Initialization
    outputs = []
    model = load_model(model_name, max_length=max_length)
    batch = initialize_batch()
    pbar = tqdm(enumerate(predicts), total=len(predicts))

    # Configuration parameters
    batch_size = config['batch_size']
    embed_threshold = config['embed_threshold']
    embed_threshold_ratio = config['embed_threshold_ratio']
    topk = len(predicts[0]['relevant'])
    
    assert topk >= output_k
    all, number = 0, 0
    # Processing
    with open(config['output_predict_txt'], 'w') as fo:
        for _, predict in pbar:
            all += 1
            if not threshold_filter(predict['score'], embed_threshold, embed_threshold_ratio):
                number += 1
                query = public_test[predict['id']]
                relevant_contexts = [corpus[context] for context in predict['relevant']]
                
                batch['queries'].extend([query] * len(relevant_contexts))
                batch['relevant_contexts'].extend(relevant_contexts)
                batch['predict_ids'].extend([predict['id']] * len(relevant_contexts))
                batch['relevant_ids'].extend(predict['relevant'])
                batch['relevant_scores'].extend(predict['score'])
            else:
                write_output(fo, predict['id'], predict['relevant'][:output_k], predict['score'], enable_filter=True)
                continue

            # Process full batch
            if len(batch['queries']) >= batch_size * len(relevant_contexts):
                process_batch(model, fo, batch)
                batch = initialize_batch()  # Clear the batch

        # Process any remaining queries
        if batch['queries']:
            process_batch(model, fo, batch, enable_filter=True)
        
        print(all, number)
    save_to_json(outputs, config['output_reranked_json'], indent=4)
    return os.path.abspath(config['output_predict_txt']), os.path.abspath(config['output_reranked_json'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference cross with configuration.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/configs/infer_cross.json", 
        help="Path to the configuration JSON file."
    )
    
    args = parser.parse_args()
    config = read_json_or_dataset(args.config_path)

    print("Config: ", json.dumps(config, indent=4, ensure_ascii=False))
    
    corpus = read_json_or_dataset(config['corpus_path'])
    predicts = read_json_or_dataset(path=config['output_detailed_predict_json'])
    output_k = config['top_k']
    corpus = {doc['id']: doc['text'] for doc in corpus}
    public_test = {test['id']: test['text'] for test in predicts}

    results = pipeline(
        model_name=config['model_name'],
        corpus=corpus,
        predicts=predicts,
        output_k=config['top_k'],
        max_length=1024
    )