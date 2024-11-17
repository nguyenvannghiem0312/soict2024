from semantic_search.inference_sbert import pipeline
from utils.io import read_json_or_dataset, save_to_json
from tqdm import tqdm

if __name__ == "__main__":
    config = {
        "corpus_path": "data/Legal Document Retrieval/corpus.json",
        "query_path": "data/Train/train.json",
        "corpus_embedding_path": "outputs/corpus_embeddings_e5.pkl",
        "output_predict_txt": "outputs/generate.txt",
        "output_predict_json": "outputs/generate.json",
        "output_detailed_predict_json": "outputs/detailed_generate.json",
        "model_name": "Turbo-AI/me5-base-v7__trim_vocab-1024",
        "top_k": 5,
        "max_length": 1022,
        "is_tokenizer": False,
        "query_prompt": "query: ",
        "corpus_prompt": "corpus: "
    }

    # Load corpus and query from JSON files
    corpus = read_json_or_dataset(config['corpus_path'])
    query = read_json_or_dataset(config['query_path'])

    corpus_dict = {item['id']: item['text'] for item in tqdm(corpus)}

    # Call the pipeline function
    _, _, detailed_results, _ = pipeline(
        model_name=config['model_name'],
        corpus=corpus,
        query=query,
        benchmark=None,
        corpus_embedding_path=config['corpus_embedding_path'],
        top_k=config['top_k'],
        is_tokenizer=config['is_tokenizer'],
        max_length=config["max_length"]
    )

    mind_hard_negatives = []

    for item in tqdm(detailed_results):
        entry = {}
        entry['id'] = item['id']
        entry['text'] = item['text']
        
        relevant_predict_ids = item['relevant']
        relevant = next((q for q in query if q['id'] == item['id']), None)['relevant']
        entry['relevant'] = relevant

        relevant_ids = [r['id'] for r in relevant if 'id' in r]

        hard_negatives = []
        for relevant_predict in relevant_predict_ids:
            if relevant_predict not in relevant_ids:
                hard_negatives.append({
                    'id': relevant_predict,
                    'text': corpus_dict[relevant_predict]
                })
        entry['not_relevant'] = hard_negatives
        mind_hard_negatives.append(entry)
    save_to_json(data=mind_hard_negatives, file_path='outputs/mind_hard_negatives.json')
    