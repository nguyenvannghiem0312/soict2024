import bm25s
import pandas as pd
import re, string
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

def all_in_one_process(sent):
    return remove_stopword(normalize_text(word_segment(clean_text(sent))))

corpus = [
    "Một con mèo là một loài động vật và thích kêu meo meo",
    "Một con chó là bạn tốt nhất của con người và thích chơi",
    "Một con chim là một loài động vật đẹp có thể bay",
    "Một con cá là một sinh vật sống dưới nước và biết bơi",
]

def splitter(text):
    return all_in_one_process(text).split()

tokenizer = Tokenizer(
        stemmer = None,
        stopwords = None,
        splitter = splitter
    )

corpus_tokens = tokenizer.tokenize(corpus)
print(corpus_tokens)
print(tokenizer.get_vocab_dict())

query = "Cá có biết kêu như mèo không?"
query_tokens = tokenizer.tokenize([query])
print(query_tokens)

retriever = bm25s.BM25()
retriever.index(corpus_tokens)

retriever.save("test_nhanh", corpus=corpus)

results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=2)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {doc}")

retriever.save("animal_index_bm25")
retriever.save("animal_index_bm25", corpus=corpus)
tokenizer.save_vocab("animal_index_bm25")
tokenizer.save_stopwords("animal_index_bm25")