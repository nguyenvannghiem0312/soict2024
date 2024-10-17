import pandas as pd
import re

def read_corpus(corpus_file):
    """corpus.csv"""
    corpus_df = pd.read_csv(corpus_file)
    corpus_list = [{'id': row['cid'], 'text': row['text']} for _, row in corpus_df.iterrows()]
    return corpus_list

def read_public_test(public_test_file):
    """public_test.csv"""
    public_test_df = pd.read_csv(public_test_file)
    public_test_list = [{'id': row['qid'], 'text': row['question']} for _, row in public_test_df.iterrows()]
    return public_test_list

def read_train(train_file, corpus_df):
    """train.csv"""
    train_df = pd.read_csv(train_file)
    train_list = []
    
    for _, row in train_df.iterrows():
        relevant_contexts = []

        row['cid'] = re.sub(r'\s+', ' ', row['cid'])
        cid_list = eval(row['cid'].strip().replace('[ ', '[').replace(' ]', ']').replace(' ', ','))
        for cid in cid_list:
            relevant_context = corpus_df[corpus_df['id'] == cid]
            if not relevant_context.empty:
                relevant_contexts.append({'id': cid, 'text': relevant_context['text'].values[0]})

        train_list.append({
            'id': row['qid'],
            'text': row['question'],
            'relevant': relevant_contexts
        })

    return train_list


