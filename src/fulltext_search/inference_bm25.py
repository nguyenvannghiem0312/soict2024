import json
from tqdm.auto import tqdm
from pyvi import ViTokenizer
from collections import Counter
from rank_bm25 import BM25Okapi
from utils.io import read_json_or_dataset, save_to_json, save_to_txt
from utils.dataset import search_by_id
import json
import argparse

def tokenize_corpus_and_save(corpus_path):
    corpus = read_json_or_dataset(corpus_path)
    tokenized_corpus = [ViTokenizer.tokenize(doc['text']).split() for doc in tqdm(corpus)]
    token_frequency = Counter(token for doc in tokenized_corpus for token in doc)

    return tokenized_corpus, token_frequency

def get_top_k_relevant_corpus(query, corpus, bm25, k=10):
    tokenized_query = ViTokenizer.tokenize(query).split()
    doc_scores = bm25.get_scores(tokenized_query)
    max_scores = max(doc_scores)
    sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]
    result = {
        'id': query,
        'text': query,
        'relevant': [corpus[i]['id'] for i in sorted_indices],
        'score': [doc_scores[i] / max_scores for i in sorted_indices]
    }
    return result

def public_test_eval(test_path, tokenized_corpus, corpus, top_k):
    test_data = read_json_or_dataset(test_path)

    bm25 = BM25Okapi(tokenized_corpus)
    
    results = []
    for item in tqdm(test_data):
        query = item["text"]
        top_k_result = get_top_k_relevant_corpus(query, corpus, bm25, k=top_k)
        results.append(top_k_result)

    return results

def negative_generation(train_path, tokenized_corpus, corpus):
    train_data = read_json_or_dataset(train_path)
    
    bm25 = BM25Okapi(tokenized_corpus)

    for item in tqdm(train_data):
        relevant_list = [rele['id'] for rele in item['relevant']]
        query = item["text"]
        top_k_result = get_top_k_relevant_corpus(query, corpus, bm25, k=10)
        negative_list = [doc_id for doc_id in top_k_result['relevant'] if doc_id not in relevant_list]
        
        item['not_relevant'] = [{'id': cid, 'text': search_by_id(data=corpus, search_id=cid)['text']} for cid in negative_list]


    return train_data

def pipeline_bm25(config):
    corpus = read_json_or_dataset(config['corpus_path'])
    try:
        tokenized_corpus = read_json_or_dataset(config['tokenizer_corpus_path'])
    except FileNotFoundError:
        tokenized_corpus, token_frequency = tokenize_corpus_and_save(config['corpus_path'])
        save_to_json(data=tokenized_corpus, file_path=config['tokenizer_corpus_path'])
        save_to_json(data=token_frequency, file_path=config['token_frequency_path'])

    if config["is_test"] == True:
        results = public_test_eval(test_path=config['query_path'], tokenized_corpus=tokenized_corpus, corpus=corpus, top_k=config["top_k"])
        save_to_json(data=results, file_path=config['result_path'])

    if config["is_generate"] == True:
        results = negative_generation(train_path=config["train_path"], tokenized_corpus=tokenized_corpus, corpus=corpus)
        save_to_json(data=results, file_path=config["generative_data_path"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BM25 pipeline with configuration.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/configs/bm25_config.json", 
        help="Path to the configuration JSON file."
    )
    
    args = parser.parse_args()
    config = read_json_or_dataset(args.config_path)

    print("Config: ", json.dumps(config, indent=4, ensure_ascii=False))
    result, mrr_score = pipeline_bm25(config)
    save_to_json(result, config['result_path'])
