import bm25s
import pickle
import numpy as np
import pandas as pd
import re, string
from pyvi.ViTokenizer import tokenize
from bm25s.tokenization import Tokenizer

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

def all_in_one_process(sent):
    return remove_stopword(normalize_text(word_segment(clean_text(sent))))

def splitter(text):
    return all_in_one_process(text).split()

def main():
    corpus_lst = [
    "Một con mèo là một loài động vật và thích kêu meo meo",
    "Một con chó là bạn tốt nhất của con người và thích chơi",
    "Một con chim là một loài động vật đẹp có thể bay",
    "Một con cá là một sinh vật sống dưới nước và biết bơi",
    ]

    queries_lst = [
        "Cá có biết kêu như mèo không?",
    ]

    # Initialize the Tokenizer with the stemmer
    tokenizer = Tokenizer(
            stemmer = None,
            stopwords = None,
            splitter = splitter
        )

    # Tokenize the corpus
    corpus_tokenized = tokenizer.tokenize(
        corpus_lst, 
        update_vocab=True, # update the vocab as we tokenize
        return_as="ids"
    )

    print(corpus_tokenized)
    print(tokenizer.get_vocab_dict())

    tokenizer_stream = tokenizer.streaming_tokenize(
        queries_lst, 
        update_vocab=False
    )

    query_ids = []

    for q in tokenizer_stream:
        # you can do something with the ids here, e.g. retrieve from the index
        if 1 in q:
            query_ids.append(q)

    # Let's see how it's all used
    retriever = bm25s.BM25()
    retriever.index(corpus_tokenized, leave_progress=False)

    # all of the above can be passed to index a bm25s model

    # e.g. using the ids directly
    results, scores = retriever.retrieve(query_ids, k=3)
    
    # or passing a tuple of ids and vocab dict
    vocab_dict = tokenizer.get_vocab_dict()
    results, scores = retriever.retrieve((query_ids, vocab_dict), k=1)

    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")


if __name__ == "__main__":
    main()
