import math
import json
import bm25s
import pickle
import numpy as np
import pandas as pd
import re, os, string
from pyvi.ViTokenizer import tokenize
from bm25s.tokenization import Tokenizer
from concurrent.futures import ProcessPoolExecutor

filename = 'stopwords.csv'
data = pd.read_csv(filename, sep="\t", encoding='utf-8')
list_stopwords = data['stopwords']

config = json.load(open('configs/bm25_config.json'))



def preprocess_data():
    with open(config['corpus_path'], 'r', encoding='utf8') as f:
        corpus_data = json.load(f)

    # Create a ProcessPoolExecutor instance with the desired number of worker processes
    with ProcessPoolExecutor(max_workers=12) as executor:
        # Use the map method to apply the all_in_one_process function to each item in corpus_data
        corpus = list(executor.map(all_in_one_process, (cor['text'] for cor in corpus_data)))

    # with open('processed_corpus.pkl', 'wb') as f:
    #     pickle.dump(corpus, f)

    tokenizer = Tokenizer(
        stemmer = None,
        stopwords = None,
        splitter = splitter
    )

    courpus_tokens = tokenizer.tokenize(corpus)
    tokenizer.save_vocab(save_dir='data/corpus_vocab')

def splitter(text):
    return text.split()

def corpus_tokenize_ready_for_bm25s():
    tokenizer = Tokenizer(
        stemmer = None,
        stopwords = None,
        splitter = splitter
    )
    with open('E:\Projects\BKAI\soict2024\src\data\processed_corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)

    courpus_tokens = tokenizer.tokenize(corpus)
    tokenizer.save_vocab(save_dir='data/corpus_vocab')

def top_k(query_lst):
    for que in query_lst:
        que = all_in_one_process(que)
    
    # Query the corpus
    new_tokenizer = Tokenizer(stemmer=None, stopwords=None, splitter=splitter)
    new_tokenizer.load_vocab("data/corpus_vocab")
    print(type(new_tokenizer))
    # print("vocab reloaded:", new_tokenizer.get_vocab_dict())
    
    query_tokens = new_tokenizer.streaming_tokenize(query_lst, update_vocab=False)
    query_ids = []
    for q in query_tokens:
        if 1 in q:
            query_ids.append(q)

    retriever = bm25s.BM25()
    retriever.index(new_tokenizer.get_vocab_dict)

    results, scores = retriever.retrieve(query_ids, k=3)
    print(results)
    print(scores)


if __name__ == '__main__':
    query_lst = [
        'Trường hợp nào được miễn lệ phí cấp hộ chiếu?',
        'Hiệp hội Công nghiệp ghi âm Việt Nam là tổ chức gì?'
    ]
    top_k(query_lst)