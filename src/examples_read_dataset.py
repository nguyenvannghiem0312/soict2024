from utils.dataset import read_corpus, read_public_test, read_train
from utils.io import save_to_json
import pandas as pd

corpus_list = read_corpus('data/Legal Document Retrieval/corpus.csv')
public_test_list = read_public_test('data/Legal Document Retrieval/public_test.csv')
train_list = read_train('data/Legal Document Retrieval/train.csv', pd.DataFrame(corpus_list))

# 
print(corpus_list[0])
print(train_list[0])
print(public_test_list[0])

save_to_json(corpus_list, 'data/Legal Document Retrieval/corpus.json')
save_to_json(train_list, 'data/Legal Document Retrieval/train.json')
save_to_json(public_test_list, 'data/Legal Document Retrieval/public_test.json')