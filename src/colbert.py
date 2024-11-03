from model2vec.distill import distill
from typing import List
from model2vec import StaticModel

import json
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def pad_sequences(sequences, dtype=torch.float32):
    """
    Pads a list of sequences (tensors) to the length of the longest sequence.
    Args:
        sequences: List of tensors, where each tensor is of shape (k, d).
    
    Returns:
        Padded tensor of shape (num_sequences, max_length, d), and a list of original lengths.
    """
    max_length = max(seq.size(0) for seq in sequences)
    padded_tensor = torch.zeros(len(sequences), max_length, sequences[0].size(1), dtype=dtype)
    lengths = []
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        padded_tensor[i, :length, :] = seq.to(dtype)
        lengths.append(length)
    return padded_tensor, lengths

def colbert_similarity_matrix(query_embeddings_list, doc_embeddings_list, dtype=torch.float32):
    """
    Computes a similarity matrix for a batch of queries and documents using the Colbert method,
    padding the queries to the maximum length.
    Args:
        query_embeddings_list: List of tensors, where each tensor is of shape (k, d) for a query with k tokens.
        doc_embeddings_list: List of tensors, where each tensor is of shape (n, d) for a document with n tokens.

    Returns:
        similarity_matrix: A tensor of shape (num_queries, num_docs) containing the similarity scores.
    """
    query_embeddings_list = [
        F.normalize(query_embedding.to(dtype), p=2, dim=1)
        for query_embedding in query_embeddings_list
    ]

    doc_embeddings_list = [
        F.normalize(doc_embedding.to(dtype), p=2, dim=1)
        for doc_embedding in doc_embeddings_list
    ]
    padded_queries, _ = pad_sequences(query_embeddings_list, dtype=dtype)
    padded_docs, _ = pad_sequences(doc_embeddings_list, dtype=dtype)    
    num_queries = padded_queries.size(0)
    num_docs = padded_docs.size(0)
    padded_queries = padded_queries.to(dtype)
    padded_docs = padded_docs.to(dtype)
    print(num_queries, num_docs, padded_queries.shape, padded_docs.shape)
    similarity_matrix = torch.zeros(num_queries, num_docs)
    
    for i in range(num_queries):
        query_embedding = padded_queries[i]
        similarity_matrix_i = torch.matmul(query_embedding, padded_docs.permute(1,2,0))
        max_sim_values = torch.max(similarity_matrix_i, dim=1)[0]  # Shape: (num_docs,)
        similarity_matrix[i] = torch.sum(max_sim_values, dim=0)

    return similarity_matrix

def colbert_similarity_matrix_no_pad(query_embeddings_list, doc_embeddings_list, dtype=torch.float32):
    """
    Computes a similarity matrix for a batch of queries and documents using the Colbert method,
    without padding the queries or documents.
    
    Args:
        query_embeddings_list: List of tensors, where each tensor is of shape (k, d) for a query with k tokens.
        doc_embeddings_list: List of tensors, where each tensor is of shape (n, d) for a document with n tokens.

    Returns:
        similarity_matrix: A tensor of shape (num_queries, num_docs) containing the similarity scores.
    """
    query_embeddings_list = [
        F.normalize(query_embedding.to(dtype), p=2, dim=1)
        for query_embedding in query_embeddings_list
    ]

    # Normalize all document embeddings
    doc_embeddings_list = [
        F.normalize(doc_embedding.to(dtype), p=2, dim=1)
        for doc_embedding in doc_embeddings_list
    ]
    num_queries = len(query_embeddings_list)
    num_docs = len(doc_embeddings_list)
    similarity_matrix = torch.zeros(num_queries, num_docs)

    for i in range(num_queries):
        query_embedding = query_embeddings_list[i].to(dtype)
        for j in range(num_docs):
            doc_embedding = doc_embeddings_list[j].to(dtype)
            similarity_matrix_i_j = torch.matmul(query_embedding, doc_embedding.T)
            max_sim_values = torch.max(similarity_matrix_i_j, dim=1)[0]  # Shape: (num_tokens_query,)
            similarity_matrix[i, j] = torch.sum(max_sim_values)
    return similarity_matrix


def get_top_k_colbert(model, queries_text: List[str], docs_text: List[str], k=10, show_progress_bar=True, dtype=torch.float32):
    query_embeddings_list = [torch.from_numpy(e) for e in model.encode_as_sequence(queries_text, show_progress_bar=show_progress_bar)]
    doc_embeddings_list = [torch.from_numpy(e) for e in model.encode_as_sequence(docs_text, show_progress_bar=show_progress_bar)]
    similarity_matrix = colbert_similarity_matrix_no_pad(query_embeddings_list, doc_embeddings_list, dtype=dtype)
    return similarity_matrix.topk(k).indices

def rerank_colbert(model, queries, corpus, k=20, show_progress_bar=True, dtype=torch.float32):
    colbert_ret = []
    for i, q in tqdm(enumerate(queries)):
        topk = get_top_k_colbert(model, 
                    [q['text'] for q in queries], 
                    [c['text'] for c in corpus[i]],
                    k=k, 
                    show_progress_bar=show_progress_bar, dtype=dtype)
        colbert_ret.append({
            'id': q['id'],
            'text': q['text'],
            'relevant': [corpus[i][j]['id'] for j in topk[i].tolist()]
        })
    return colbert_ret

if __name__ == '__main__':
    model_name = "Turbo-AI/me5-base-v3__trim-vocab"
    m2v_model = distill(model_name=model_name, pca_dims=256)
    m2v_model.save_pretrained("m2v_model")
    smodel = StaticModel.from_pretrained("m2v_model")
    queries = json.load(open('data/Dev/query_dev.json'))
    corpus = json.load(open('data/Dev/corpus_dev.json'))
    cross = json.load(open('data/Dev/cross_dev.json'))
    corpus_dict = {c['id']:c['text'] if type(c['text']) != dict else c['text']['text'] for c in corpus}
    ans = rerank_colbert(smodel,
        cross,
        [[
            {
                'id': l,
                'text': corpus_dict[l]
            }
            for l in cr['relevant']
        ] for cr in cross],
        k=20, show_progress_bar=False, dtype=torch.float32
    )
    json.dump(ans, open("data/colbert.json", "w"), indent=4)