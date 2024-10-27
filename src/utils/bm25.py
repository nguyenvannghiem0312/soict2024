import json
import pickle
import py_vncorenlp
from tqdm.auto import tqdm
from pyvi import ViTokenizer
from collections import Counter
from rank_bm25 import BM25Okapi
from multiprocessing import Pool

with open('configs/infer_sbert.json', 'r') as f:
    config = json.load(f)

with open('configs/sbert.json', 'r') as f:
    sbert_config = json.load(f)


corpus_path = config['corpus_path']
query_path = config['query_path']
train_path = sbert_config['train_path']
# train_path = 'data/Legal Document Retrieval/public_test.json'

def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data

def tokenize_corpus_and_save(corpus_path):
    corpus = read_json(corpus_path)
    tokenized_corpus = [ViTokenizer.tokenize(doc['text']).split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    token_frequency = Counter(token for doc in tokenized_corpus for token in doc)

    with open('data/Legal Document Retrieval/tokenized_corpus.pkl', 'wb') as f:
        pickle.dump(tokenized_corpus, f)
    with open('data/Legal Document Retrieval/bm25_model.pkl', 'wb') as f:
        pickle.dump(bm25, f)
    with open('data/Legal Document Retrieval/token_frequency.json', 'w') as f:
        json.dump(token_frequency, f)

def get_10_most_relevant_corpus_id(query, bm25_model_path):
    with open(bm25_model_path, 'rb') as f:
        bm25 = pickle.load(f)
    corpus = read_json(corpus_path)
    tokenized_query = ViTokenizer.tokenize(query).split()
    doc_scores = bm25.get_scores(tokenized_query)
    sorted_corpus_id = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:10]
    top_10_ids = [corpus[i]['id'] for i in sorted_corpus_id]
    return top_10_ids

def public_test_eval(bm25_model_path):
    with open(bm25_model_path, 'rb') as f:
        bm25 = pickle.load(f)
    test_data = read_json(train_path)
    corpus = read_json(corpus_path)
    with open('data/similar_corpus_ids.txt', 'w') as output_file:
        for item in tqdm(test_data):
            query = item["text"]
            tokenized_query = ViTokenizer.tokenize(query).split()
            scores = bm25.get_scores(tokenized_query)
            top_10_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
            top_10_ids = [str(corpus[i]['id']) for i in top_10_indices]
            output_file.write(" ".join(top_10_ids) + '\n')

def process_query(item, bm25_model_path):
    with open(bm25_model_path, 'rb') as f:
        bm25 = pickle.load(f)
    corpus = read_json(corpus_path)
    query = item["text"]
    tokenized_query = ViTokenizer.tokenize(query).split()
    scores = bm25.get_scores(tokenized_query)
    top_10_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
    top_10_ids = [str(corpus[i]['id']) for i in top_10_indices]
    return " ".join(top_10_ids)

def public_test_eval_parallel():
    test_data = read_json(train_path)
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_query, test_data), total=len(test_data)))

    with open('data/similar_corpus_ids.txt', 'w') as output_file:
        output_file.write("\n".join(results))

def negative_generation(bm25_model_path):
    train_data = read_json(train_path)
    corpus = read_json(corpus_path)
    with open(bm25_model_path, 'rb') as f:
        bm25 = pickle.load(f)
    negative_data = []
    for item in tqdm(train_data):
        relevant_list = [rele['id'] for rele in item['relevant']]
        query = item["text"]
        tokenized_query = ViTokenizer.tokenize(query).split()
        scores = bm25.get_scores(tokenized_query)
        top_10_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        top_10_ids = [corpus[i]['id'] for i in top_10_indices]
        negative_list = [doc_id for doc_id in top_10_ids if doc_id not in relevant_list]
        negative_data.append({
            "id": item["id"],
            "negative": negative_list[0]
        })
    with open('data/negative_data.json', 'w') as f:
        json.dump(negative_data, f)

if __name__ == '__main__':
    # tokenize_corpus_and_save(corpus_path)

    # query = "Nội dung lồng ghép vấn đề bình đẳng giới trong xây dựng văn bản quy phạm pháp luật được quy định thế nào?"
    # print(get_10_most_relevant_corpus_id(query, 'data/Legal Document Retrieval/bm25_model.pkl'))
    public_test_eval('data/bm25_model.pkl')