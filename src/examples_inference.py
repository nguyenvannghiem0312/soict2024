import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from utils.io import save_to_txt
import warnings
warnings.filterwarnings("ignore")

def load_model(model_name='distilbert-base-nli-mean-tokens'):
    return SentenceTransformer(model_name)

def encode_corpus(model: SentenceTransformer, corpus, corpus_embedding_path='outputs/corpus_embeddings.pkl'):
    corpus_texts = [entry['text'] for entry in corpus]
    corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)
    corpus_embeddings = corpus_embeddings.cpu()
    with open(corpus_embedding_path, 'wb') as f:
        pickle.dump((corpus, corpus_embeddings), f)
    print(f'Corpus embeddings saved to {corpus_embedding_path}')

def load_corpus(corpus_embedding_path='corpus_embeddings.pkl'):
    with open(corpus_embedding_path, 'rb') as f:
        corpus, corpus_embeddings = pickle.load(f)
    print(f'Corpus embeddings loaded from {corpus_embedding_path}')
    return corpus, corpus_embeddings

def retrieve(model, corpus, corpus_embeddings, query, top_k=10):
    query_texts = [entry['text'] for entry in query]
    query_embeddings = model.encode(query_texts, convert_to_tensor=True)
    query_embeddings = query_embeddings.cpu()

    results = []
    for query_idx, query_embedding in tqdm(enumerate(query_embeddings)):
        cosine_scores = torch.cosine_similarity(query_embedding.unsqueeze(0), corpus_embeddings, dim=1)
        top_results = torch.topk(cosine_scores, k=top_k)

        result_entry = [query[query_idx]['id']]
        for score_idx in top_results[1]:
            result_entry.append(corpus[score_idx]['id'])
        results.append(result_entry)
    
    return results

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

def pipeline(model_name, corpus, query, benchmark, corpus_embedding_path='outputs/corpus_embeddings.pkl', top_k=10):
    model = load_model(model_name)
    
    # Check if corpus embeddings file exists
    try:
        corpus, corpus_embeddings = load_corpus(corpus_embedding_path)
    except FileNotFoundError:
        encode_corpus(model, corpus, corpus_embedding_path)
        corpus, corpus_embeddings = load_corpus(corpus_embedding_path)
    
    # Retrieve results
    results = retrieve(model, corpus, corpus_embeddings, query, top_k)

    # Calculate MRR
    if benchmark:
        mrr_score = calculate_mrr(results, benchmark)
    else:
        mrr_score = 0

    return results, mrr_score



if __name__ == '__main__':
    corpus = [
        {'id': 'context1', 'text': 'This is the first context.'},
        {'id': 'context2', 'text': 'Another relevant context goes here.'},
    ]
    
    query = [
        {'id': 'query1', 'text': 'Find relevant information for this query.'},
        {'id': 'query2', 'text': 'Another query to match.'},
    ]
    
    benchmark = {
        'query1': ['context1'],
        'query2': ['context2'],
    }

    results, mrr_score = pipeline(
        model_name = 'distilbert-base-nli-mean-tokens', 
        corpus = corpus,
        query = query,
        benchmark = benchmark, 
        corpus_embedding_path='outputs/corpus_embeddings.pkl', 
        top_k=1
    )

    print("MRR score: ", mrr_score)
    save_to_txt(data=results, file_path='outputs/predict.txt')
