from tqdm import tqdm
from utils.io import read_json, save_to_json, save_to_txt, convert_results
from fulltext_search.inference_bm25s import pipeline_bm25
from semantic_search.inference_sbert import pipeline

def combine_bm25_embedding_results(bm25_results, embedding_results, ratio=(0.3, 0.7), top_k=10):
    combined_results = []

    for query in tqdm(bm25_results):
        query_id = query['id']
        bm25_docs = query['relevant']
        bm25_scores = query['score']
        
        emb_query = next((item for item in embedding_results if item['id'] == query_id), None)
        if emb_query:
            emb_docs = emb_query['relevant']
            emb_scores = emb_query['score']
        else:
            emb_docs, emb_scores = [], []
        
        bm25_dict = {doc_id: score for doc_id, score in zip(bm25_docs, bm25_scores)}
        emb_dict = {doc_id: score for doc_id, score in zip(emb_docs, emb_scores)}
        
        combined_dict = {}
        for doc_id in bm25_dict:
            if doc_id in emb_dict:
                combined_score = ratio[0] * bm25_dict[doc_id] + ratio[1] * emb_dict[doc_id]
                combined_dict[doc_id] = combined_score
        
        combined_sorted = sorted(combined_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        combined_results.append(
            {
            'id': query_id,
            'text': query['text'],
            'relevant': [doc_id for doc_id, _ in combined_sorted],
            'score': [score for _, score in combined_sorted]
            }
        )

    return combined_results


def combined_pipeline(config, benchmark):
    bm25_results, _ = pipeline_bm25(config)
    
    queries = read_json(config['query_path'])
    corpus = read_json(config['corpus_path'])
    model_name = config['model_name']
    embedding_results, _, detailed_embedding_results, _ = pipeline(
                                                            model_name=model_name,
                                                            corpus=corpus,
                                                            query=queries,
                                                            benchmark=benchmark,
                                                            corpus_embedding_path=config["corpus_embedding_path"],
                                                            top_k=config['top_k'],
                                                            is_tokenizer=config['is_tokenizer']
                                                        )
    
    combined_results = combine_bm25_embedding_results(
                        bm25_results,
                        detailed_embedding_results,
                        ratio=(config['bm25_weight'], config['emb_weight']),
                        top_k=config['output_top_k']
                        )
    
    return bm25_results, detailed_embedding_results, combined_results

def calculate_mrr(benchmark, combined_results, top_k=10):
    total_rr = 0.0
    query_count = 0

    for query in benchmark:
        query_id = query['id']
        relevant_ids = {doc['id'] for doc in query['relevant']}

        combined_query_result = combined_results.get(query_id, {})
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

def find_best_combination_ratio(benchmark, bm25_results, embedding_results, top_k=10, step=0.1):
    best_ratio = (0, 1)
    best_mrr = 0

    for bm25_weight in tqdm(range(0, int(1/step) + 1)):
        bm25_ratio = bm25_weight * step
        embedding_ratio = 1 - bm25_ratio
        
        combined_results = combine_bm25_embedding_results(
            bm25_results,
            embedding_results,
            ratio=(bm25_ratio, embedding_ratio),
            top_k=top_k
        )
        
        mrr_score = calculate_mrr(benchmark, combined_results, top_k=top_k)
        
        if mrr_score > best_mrr:
            best_mrr = mrr_score
            best_ratio = (bm25_ratio, embedding_ratio)

    return best_ratio, best_mrr


if __name__ == '__main__':
    config = read_json('configs/hybrid_config.json')

    benchmark = read_json(config['benchmark'])
    bm25_results, embedding_results, combined_results = combined_pipeline(
        config=config,
        benchmark=None
    )
    save_to_json(combined_results, 'outputs/predict.json')
    save_to_txt(convert_results(combined_results), 'outputs/predict.txt')

    mrr_score = calculate_mrr(
        benchmark=benchmark,
        combined_results=combined_results
    )
    print(mrr_score)


    best_ratio, best_mrr = find_best_combination_ratio(
        benchmark=benchmark,
        bm25_results=bm25_results,
        embedding_results=embedding_results,
        top_k=10,
        step=0.05
    )

    print(best_ratio, best_mrr)