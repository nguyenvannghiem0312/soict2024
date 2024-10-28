import json
from tqdm.auto import tqdm
from pyvi import ViTokenizer
from collections import Counter
from rank_bm25 import BM25Okapi
from utils.io import read_json, save_to_json, save_to_txt
from utils.dataset import search_by_id

def tokenize_corpus_and_save(corpus_path):
    corpus = read_json(corpus_path)
    tokenized_corpus = [ViTokenizer.tokenize(doc['text']).split() for doc in tqdm(corpus)]
    token_frequency = Counter(token for doc in tokenized_corpus for token in doc)

    return tokenized_corpus, token_frequency

def get_top_k_relevant_corpus(query, corpus, bm25, k=10):
    tokenized_query = ViTokenizer.tokenize(query).split()
    doc_scores = bm25.get_scores(tokenized_query)
    normalized_scores = [score / max(doc_scores) for score in doc_scores] if doc_scores else []
    sorted_indices = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i], reverse=True)[:k]
    result = {
        'id': query,
        'text': query,
        'relevant': [corpus[i]['id'] for i in sorted_indices],
        'score': [normalized_scores[i] for i in sorted_indices]
    }
    return result

def public_test_eval(test_path, tokenized_corpus, corpus, top_k):
    test_data = read_json(test_path)

    bm25 = BM25Okapi(tokenized_corpus)
    
    results = []
    for item in tqdm(test_data):
        query = item["text"]
        top_k_result = get_top_k_relevant_corpus(query, corpus, bm25, k=top_k)
        results.append(top_k_result)

    return results

def negative_generation(train_path, tokenized_corpus, corpus):
    train_data = read_json(train_path)
    
    bm25 = BM25Okapi(tokenized_corpus)

    for item in tqdm(train_data):
        relevant_list = [rele['id'] for rele in item['relevant']]
        query = item["text"]
        top_k_result = get_top_k_relevant_corpus(query, corpus, bm25, k=10)
        negative_list = [doc_id for doc_id in top_k_result['relevant'] if doc_id not in relevant_list]
        
        item['not_relevant'] = [{'id': cid, 'text': search_by_id(data=corpus, search_id=cid)} for cid in negative_list]


    return train_data

def main():
    config = read_json('configs/bm25_config.json')
    corpus = read_json(config['corpus_path'])
    try:
        tokenized_corpus = read_json(config['tokenizer_corpus_path'])
    except FileNotFoundError:
        tokenized_corpus, token_frequency = tokenize_corpus_and_save(config['corpus_path'])
        save_to_json(data=tokenized_corpus, file_path='data/Legal Document Retrieval/tokenized_corpus.json')
        save_to_json(data=token_frequency, file_path='data/Legal Document Retrieval/token_frequency.json')

    if config["is_test"] == True:
        results = public_test_eval(test_path=config['query_path'], tokenized_corpus=tokenized_corpus, corpus=corpus, top_k=config["top_k"])
        save_to_json(data=results, file_path=config['result_path'])

    if config["is_generate"] == True:
        results = negative_generation(train_path=config["train_path"], tokenized_corpus=tokenized_corpus, corpus=corpus)
        save_to_json(data=results, file_path=config["generative_data_path"])

if __name__ == '__main__':
    main()
