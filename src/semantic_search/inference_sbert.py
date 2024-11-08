import pickle
import numpy as np
import argparse
import json
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import torch
from tqdm import tqdm
from utils.io import save_to_txt, read_json_or_dataset, save_to_json
from utils.model import load_model2vec
import warnings
warnings.filterwarnings("ignore")

def load_model(model_name='Turbo-AI/me5-base-v3__trim-vocab', max_length=512):
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_length
    return model

def encode_corpus(model: SentenceTransformer, corpus, corpus_embedding_path='outputs/corpus_embeddings.pkl', is_tokenizer=True, corpus_prompt=""):
    if is_tokenizer == True:
        corpus_texts = [tokenize(corpus_prompt + entry['text']) for entry in tqdm(corpus)]
    else:
        corpus_texts = [corpus_prompt + entry['text'] for entry in corpus]
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

def retrieve(model, corpus, corpus_embeddings, query, top_k=10, batch_size=32, is_tokenizer=True, query_prompt=""):
    if is_tokenizer:
        query_texts = [tokenize(query_prompt + entry['text']) for entry in query]
    else:
        query_texts = [query_prompt + entry['text'] for entry in query]

    # Encode query texts in batches
    query_embeddings = model.encode(query_texts, convert_to_tensor=True, batch_size=batch_size)
    query_embeddings = query_embeddings.cpu()

    results = []
    results_json = []
    detailed_results = []  # New list for the detailed output

    for batch_start in tqdm(range(0, len(query_embeddings), batch_size)):
        batch_query_embeddings = query_embeddings[batch_start:batch_start + batch_size]

        # Compute cosine similarity for the whole batch at once
        batch_query_embeddings_norm = torch.nn.functional.normalize(batch_query_embeddings, p=2, dim=1)
        corpus_embeddings_norm = torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1)
        cosine_scores_batch = torch.mm(batch_query_embeddings_norm, corpus_embeddings_norm.T)

        # Process each query in the batch
        for query_idx, cosine_scores in enumerate(cosine_scores_batch):
            top_results = torch.topk(cosine_scores, k=top_k)

            # Prepare the first output entry
            result_entry = [query[batch_start + query_idx]['id']]  # Add query id
            for score_idx in top_results[1]:
                result_entry.append(corpus[score_idx]['id'])  # Add top context ids
            results.append(result_entry)

            # Prepare the second output entry
            result_json_entry = {
                'id': query[batch_start + query_idx]['id'],
                'relevant': [corpus[score_idx]['id'] for score_idx in top_results[1]]
            }
            results_json.append(result_json_entry)

            # Prepare the detailed output entry
            query_id = query[batch_start + query_idx]['id']
            query_text = query[batch_start + query_idx]['text']
            relevant_ids = [corpus[score_idx]['id'] for score_idx in top_results[1]]
            scores = top_results[0].tolist()  # Convert to list for JSON serialization

            detailed_result_entry = {
                'id': query_id,
                'text': query_text,
                'relevant': relevant_ids,
                'score': scores
            }
            detailed_results.append(detailed_result_entry)

    return results, results_json, detailed_results

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

def pipeline(model_name, corpus, query, benchmark, corpus_embedding_path='outputs/corpus_embeddings.pkl', top_k=10, is_tokenizer=True, max_length=512, query_prompt="", corpus_prompt=""):
    model = load_model(model_name, max_length)
    
    # Check if corpus embeddings file exists
    try:
        corpus, corpus_embeddings = load_corpus(corpus_embedding_path)
    except FileNotFoundError:
        encode_corpus(model, corpus, corpus_embedding_path, is_tokenizer=is_tokenizer, corpus_prompt=corpus_prompt)
        corpus, corpus_embeddings = load_corpus(corpus_embedding_path)
    
    # Retrieve results
    results, results_json, detailed_results = retrieve(model, corpus, corpus_embeddings, query, top_k=top_k, is_tokenizer=is_tokenizer, query_prompt=query_prompt)

    # Calculate MRR
    if benchmark:
        mrr_score = calculate_mrr(results, benchmark)
    else:
        mrr_score = 0

    return results, results_json, detailed_results, mrr_score



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference SBERT with configuration.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/configs/infer_sbert.json", 
        help="Path to the configuration JSON file."
    )
    
    args = parser.parse_args()
    config = read_json_or_dataset(args.config_path)

    print("Config: ", json.dumps(config, indent=4, ensure_ascii=False))
    
    # Load corpus and query from JSON files
    corpus = read_json_or_dataset(config['corpus_path'])
    query = read_json_or_dataset(config['query_path'])

    # Call the pipeline function
    results, results_json, detailed_results, mrr_score = pipeline(
        model_name=config['model_name'],
        corpus=corpus,
        query=query,
        benchmark=None,
        corpus_embedding_path=config['corpus_embedding_path'],
        top_k=config['top_k'],
        is_tokenizer=config['is_tokenizer'],
        max_length=config["max_length"]
    )

    save_to_txt(data=results, file_path=config['output_predict_txt'])
    save_to_json(data=results_json, file_path=config['output_predict_json'])
    save_to_json(data=detailed_results, file_path=config['output_detailed_predict_json'])