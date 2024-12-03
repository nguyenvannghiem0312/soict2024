from utils.dataset import read_corpus, read_public_test, read_train
from utils.io import save_to_json, save_data_to_hub
import pandas as pd

corpus_list = read_corpus('data/Legal Document Retrieval/corpus.csv')
public_test_list = read_public_test('data/Legal Document Retrieval/public_test.csv')
private_test_litst = read_public_test('data/Legal Document Retrieval/private_test.csv')
train_list = read_train('data/Legal Document Retrieval/train.csv', pd.DataFrame(corpus_list))

# 
print(corpus_list[0])
print(train_list[0])
print(public_test_list[0])
print(private_test_litst[0])

save_data_to_hub(corpus_list, 'Turbo-AI/data-corpus')
save_data_to_hub(train_list, 'Turbo-AI/data-train')
save_data_to_hub(public_test_list, 'Turbo-AI/data-public_test')
save_data_to_hub(private_test_litst, 'Turbo-AI/private-test')