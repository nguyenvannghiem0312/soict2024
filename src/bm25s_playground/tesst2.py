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

def clean_text(text):
    text = re.sub('<.*?>', '', text).strip()
    text = re.sub('(\s)+', r'\1', text)
    return text

def normalize_text(text):
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()

def remove_stopword(text):
    pre_text = []
    words = text.split()
    for word in words:
        if word not in list_stopwords:
            pre_text.append(word)
    text2 = ' '.join(pre_text)

    return text2

def word_segment(sent):
    sent = tokenize(sent.encode('utf-8').decode('utf-8'))
    return sent

def splitter(text):
    return remove_stopword(normalize_text(word_segment(clean_text(text)))).split()

tokenizer = Tokenizer(
        stemmer = None,
        stopwords = None,
        splitter = splitter
    )

queries_lst = ["Cá có biết kêu như mèo không?"]

tokenizer_stream = tokenizer.streaming_tokenize(
    queries_lst,
    update_vocab=False
)

queries_tokenized  = bm25s.tokenize(queries_lst, stemmer=None, return_ids=False)

corpus = [
    "Một con mèo là một loài động vật và thích kêu meo meo",
    "Một con chó là bạn tốt nhất của con người và thích chơi",
    "Một con chim là một loài động vật đẹp có thể bay",
    "Một con cá là một sinh vật sống dưới nước và biết bơi",
]

retriever = bm25s.BM25.load('animal_index_bm25', mmap=True, load_corpus=True)

results = retriever.retrieve(queries_tokenized, k=2)

print(results)