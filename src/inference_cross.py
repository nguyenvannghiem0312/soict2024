import logging
import warnings
import os
from tqdm.auto import tqdm
from utils.io import read_json, save_to_json

from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

def load_model(model_name='Alibaba-NLP/gte-multilingual-reranker-base', max_length=512):
    model = CrossEncoder(model_name=model_name, trust_remote_code=True, max_length=max_length)
    model.model.to('cuda')
    return model

def pipeline(model_name, corpus, predicts, output_k=10, max_length=512):
    batched_queries = []
    batched_relevant_contexts = []
    batched_predict_ids = []
    batched_relevant_ids = []
    batched_relevant_scores = []
    outputs = []

    topk = len(predicts[0]['relevant'])

    assert topk >= output_k

    def write_to_txt(fo, i, enable_filter=False):
        predict_id = batched_predict_ids[i * topk]
        relevant = batched_relevant_ids[i * topk:(i + 1) * topk]
        score_slice = scores[i * topk:(i + 1) * topk]
        sorted_results = sorted(zip(score_slice, relevant), key=lambda x: -x[0])[:output_k]
        if enable_filter:
            if ((batched_relevant_scores[i * topk] - batched_relevant_scores[i * topk + 1] >= embed_threshold)
                    and ((batched_relevant_scores[i * topk] - batched_relevant_scores[i * topk + 1]) /
                        (batched_relevant_scores[i * topk + 1] - batched_relevant_scores[i * topk + 2]) >= embed_threshold_ratio)):
                sorted_results = zip(score_slice[:output_k], relevant[:output_k])
        fo.write(str(predict_id) + ' ' + ' '.join([str(x[1]) for x in sorted_results]) + '\n')
        outputs.append({
            'id': predict_id,
            'text': public_test[predict_id],
            'relevant': [x[1] for x in sorted_results],
        })

    def threshold_filter(scores, threshold=embed_threshold, ratio=embed_threshold_ratio):
        return ((scores[0] - scores[1] >= threshold) and ((scores[0] - scores[1]) / (scores[1] - scores[2]) >= ratio))

    def extend_batch(query, relevant_contexts, predict):
        batched_queries.extend([query] * len(relevant_contexts))
        batched_relevant_contexts.extend(relevant_contexts)
        batched_predict_ids.extend([predict['id']] * len(relevant_contexts))
        batched_relevant_ids.extend(predict['relevant'])
        batched_relevant_scores.extend(predict['score'])

    def clear_batch():
        batched_queries.clear()
        batched_relevant_contexts.clear()
        batched_predict_ids.clear()
        batched_relevant_ids.clear()
        batched_relevant_scores.clear()

    model = load_model(model_name, max_length=max_length)
    pbar = tqdm(enumerate(predicts), total=len(predicts))
    batch_size = config['batch_size']
    embed_threshold = config['embed_threshold']
    embed_threshold_ratio = config['embed_threshold_ratio']
    
    with open(config['output_predict_txt'], 'w') as fo:
        for k, predict in pbar:
            if threshold_filter(predict['score']) == False:
                query = public_test[predict['id']]
                relevant_contexts = [corpus[context] for context in predict['relevant']]
                extend_batch(query, relevant_contexts, predict)
            else:
                fo.write(str(predict['id']) + ' ' + ' '.join([str(x) for x in predict['relevant'][:output_k]]) + '\n')
                outputs.append({
                    'id': predict['id'],
                    'text': public_test[predict['id']],
                    'relevant': predict['relevant'][:output_k],
                })
                continue

            if len(batched_queries) == batch_size * len(relevant_contexts):
                scores = model.predict(list(zip(batched_queries, batched_relevant_contexts)), batch_size=batch_size)
                assert len(scores) == batch_size * topk
                for i in range(batch_size):
                    write_to_txt(fo, i)
                clear_batch()

        # Process any remaining queries
        if batched_queries:
            scores = model.predict(list(zip(batched_queries, batched_relevant_contexts)))
            for i in range(len(batched_queries) // topk):
                write_to_txt(fo, i, enable_filter=True)
        save_to_json(outputs, config['output_reranked_json'], indent=4)
        return os.path.abspath(config['output_predict_txt']), os.path.abspath(config['output_reranked_json'])

if __name__ == '__main__':
    config = read_json(path="configs/infer_cross.json")
    corpus = read_json(config['corpus_path'])
    predicts = read_json(path=config['output_detailed_predict_json'])
    output_k = config['top_k']
    corpus = {doc['id']: doc['text'] for doc in corpus}
    public_test = {test['id']: test['text'] for test in predicts}

    results = pipeline(
        model_name=config['model_name'],
        corpus=corpus,
        predicts=predicts,
        output_k=config['top_k'],
        max_length=config['max_length']
    )