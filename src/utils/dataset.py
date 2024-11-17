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

def search_by_id(data, search_id):
    for item in data:
        if item.get('id') == search_id:
            return item
    return None


def process_data(train, number_negatives=3):
    output = []
    for item in train:
        query_text = item['text']
        
        for relevant_context in item['relevant']:
            entry = {
                "anchor": query_text,
                "positive": relevant_context['text']
            }
            
            if 'not_relevant' in item and len(item['not_relevant']) >= number_negatives and number_negatives != 0:
                negative_entry = entry.copy()
                for idx, irrelevant_context in enumerate(item['not_relevant'][:number_negatives]):
                    negative_entry[f"negative_{idx}"] = irrelevant_context['text']
                output.append(negative_entry)
            if number_negatives == 0:
                output.append(entry)
    
    return output

def process_dev(corpus, query, query_prompt="", corpus_prompt=""):
    queries = {}
    corpus_dict = {}
    relevant_docs = {}

    for q in query:
        qid = str(q['id'])
        queries[qid] = query_prompt + q['text']
        
        # Xử lý relevant docs cho từng query
        relevant_set = set()
        for relevant in q['relevant']:
            relevant_set.add(str(relevant['id']))
        
        relevant_docs[qid] = relevant_set

    for c in corpus:
        cid = str(c['id'])
        corpus_dict[cid] = corpus_prompt + c['text']

    return {
        "queries": queries,
        "corpus": corpus_dict,
        "relevant_docs": relevant_docs
    }
