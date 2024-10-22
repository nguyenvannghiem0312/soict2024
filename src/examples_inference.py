import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pyvi.ViTokenizer import tokenize
import torch
from tqdm import tqdm
from utils.io import save_to_txt, read_json, save_to_json
from utils.dataset import search_by_id
import warnings
warnings.filterwarnings("ignore")

def load_model(model_name='NghiemAbe/Vi-Legal-Bi-Encoder-v2'):
    return SentenceTransformer(model_name, trust_remote_code=True)

def encode_corpus(model: SentenceTransformer, corpus, corpus_embedding_path='outputs/corpus_embeddings.pkl', is_tokenizer=True):
    if is_tokenizer == True:
        corpus_texts = [tokenize(entry['text']) for entry in tqdm(corpus)]
    else:
        corpus_texts = [entry['text'] for entry in corpus]
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True, batch_size=32)
    corpus_embeddings = corpus_embeddings.cpu()
    with open(corpus_embedding_path, 'wb') as f:
        pickle.dump((corpus, corpus_embeddings), f)
    print(f'Corpus embeddings saved to {corpus_embedding_path}')

def load_corpus(corpus_embedding_path='corpus_embeddings.pkl'):
    with open(corpus_embedding_path, 'rb') as f:
        corpus, corpus_embeddings = pickle.load(f)
    print(f'Corpus embeddings loaded from {corpus_embedding_path}')
    return corpus, corpus_embeddings

def retrieve(model, corpus, corpus_embeddings, query, top_k=10, batch_size=32, is_tokenizer=True):
    if is_tokenizer == True:
        query_texts = [tokenize(entry['text']) for entry in query]
    else:
        query_texts = [entry['text'] for entry in query]
    # Encode query texts in batches
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, batch_size=batch_size)
    query_embeddings = query_embeddings.cpu()

    results = []
    results_json = []
    for batch_start in tqdm(range(0, len(query_embeddings), batch_size)):
        batch_query_embeddings = query_embeddings[batch_start:batch_start + batch_size]

        # Compute cosine similarity for the whole batch at once
        batch_query_embeddings_norm = torch.nn.functional.normalize(batch_query_embeddings, p=2, dim=1)
        corpus_embeddings_norm = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)
        cosine_scores_batch = torch.mm(batch_query_embeddings_norm, corpus_embeddings_norm.T)

        # Process each query in the batch
        for query_idx, cosine_scores in enumerate(cosine_scores_batch):
            top_results = torch.topk(cosine_scores, k=top_k)

            result_entry = [query[batch_start + query_idx]['id']]  # Add query id
            result_json_entry = {
                'id': query[batch_start + query_idx]['id'],
                'relevant': []
            }
            for score_idx in top_results[1]:
                result_entry.append(corpus[score_idx]['id'])  # Add top context ids
                # Add relevant contexts to all_results
                result_json_entry['relevant'].append(corpus[score_idx]['id'])
            results.append(result_entry)
            results_json.append(result_json_entry)

    return results, results_json

def calculate_mrr(results, benchmark):
    mrr_total = 0
    for query_result in results:
        query_id = query_result[0]
        relevant_contexts = benchmark.get(query_id, [])
        for rank, retrieved_id in enumerate(query_result[1:], start=1):
            if retrieved_id in relevant_contexts:
                mrr_total += 1 / rank
                break

    return mrr_total / len(results)

def pipeline(model_name, corpus, query, benchmark, corpus_embedding_path='outputs/corpus_embeddings.pkl', top_k=10, is_tokenizer=True):
    model = load_model(model_name)
    
    # Check if corpus embeddings file exists
    try:
        corpus, corpus_embeddings = load_corpus(corpus_embedding_path)
    except FileNotFoundError:
        encode_corpus(model, corpus, corpus_embedding_path, is_tokenizer=is_tokenizer)
        corpus, corpus_embeddings = load_corpus(corpus_embedding_path)
    
    # Retrieve results
    results, results_json = retrieve(model, corpus, corpus_embeddings, query, top_k=top_k, is_tokenizer=is_tokenizer)

    # Calculate MRR
    if benchmark:
        mrr_score = calculate_mrr(results, benchmark)
    else:
        mrr_score = 0

    return results, results_json, mrr_score



if __name__ == '__main__':
    corpus = [
        {'id': 'context1', 'text': 'This is the first context.'},
        {'id': 'context2', 'text': 'Another relevant context goes here.'},
    ]
    
    corpus = read_json('data/Legal Document Retrieval/corpus.json')
    query = [
        {'id': 'query1', 'text': 'Find relevant information for this query.'},
        {'id': 'query2', 'text': 'Another query to match.'},
    ]
    query = read_json('data/Legal Document Retrieval/public_test.json')
    benchmark = {
        'query1': ['context1'],
        'query2': ['context2'],
    }

    results, results_json, mrr_score = pipeline(
        model_name = 'intfloat/multilingual-e5-large', 
        corpus = corpus,
        query = query,
        benchmark = None, 
        corpus_embedding_path='outputs/corpus_embeddings_e5_large.pkl', 
        top_k=20,
        is_tokenizer=False
    )

    print(results)

    print("MRR score: ", mrr_score)
    save_to_txt(data=results, file_path='outputs/predict.txt')
    save_to_json(data=results_json, file_path='outputs/predict.json')
