import json
from tqdm import tqdm
from utils.io import read_json_or_dataset, convert_results, save_to_txt

def combine_scores(list1, list2, a, b, top_k=10):
    combined_list = []
    
    for item1 in tqdm(list1):
        query_id = item1['id']
        item2 = next((item for item in list2 if item['id'] == query_id), None)
        
        
        contexts_scores_1 = {context: score for context, score in zip(item1['relevant'], item1['score'])}
        contexts_scores_2 = {context: score for context, score in zip(item2['relevant'], item2['score'])}

        combined_scores = []

        for id_context in item1['relevant']:
            combined_scores.append({
                'id': id_context,
                'score': a * contexts_scores_1[id_context] + b * contexts_scores_2[id_context]
            })

        combined_scores.sort(key=lambda x: x["score"], reverse=True)  # Sắp xếp giảm dần theo score

        combined_item = {
            "id": item1["id"],
            "text": item1["text"],
            "relevant": [entry["id"] for entry in combined_scores[:top_k]],
            "score": [entry["score"] for entry in combined_scores[:top_k]]
        }

        combined_list.append(combined_item)
    
    return combined_list

def calculate_mrr(benchmark, combined_results, top_k=10):
    total_rr = 0.0
    query_count = 0

    for query in benchmark:
        query_id = query['id']
        relevant_ids = {doc['id'] for doc in query['relevant']}

        combined_query_result = next((item for item in combined_results if item['id'] == query_id), None)
        ranked_docs = combined_query_result.get('relevant', [])[:top_k]

        rr = 0.0
        for rank, doc_id in enumerate(ranked_docs, start=1):
            if doc_id in relevant_ids:
                rr = 1.0 / rank
                break

        total_rr += rr
        query_count += 1

    mrr_score = total_rr / query_count if query_count > 0 else 0
    return mrr_score

def find_best_combination_ratio(benchmark, ranker_results, embedding_results, top_k=10, step=0.1):
    best_ratio = (0, 1)
    best_mrr = 0

    for ranker_weight in tqdm(range(0, int(1/step) + 1)):
        ranker_ratio = ranker_weight * step
        embedding_ratio = 1 - ranker_ratio
        
        combined_results = combined_results = combine_scores(
                ranker_results,
                embedding_results,
                ranker_results,
                a=ranker_ratio,
                b=embedding_ratio,
            )
        
        mrr_score = calculate_mrr(benchmark, combined_results, top_k=top_k)
        
        if mrr_score > best_mrr:
            best_mrr = mrr_score
            best_ratio = (ranker_ratio, embedding_ratio)

    return best_ratio, best_mrr



gte_predict = read_json_or_dataset('outputs/detailed_predict_gte.json')
ranker_predict = read_json_or_dataset('outputs/predict_reranked.json')


combine_predict = combine_scores(ranker_predict, gte_predict, a = 0.2, b = 0.8)
resutls = convert_results(combine_predict)
save_to_txt(resutls, file_path='submissions/predict.txt')


