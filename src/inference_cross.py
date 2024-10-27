import logging
import warnings
import os
from tqdm.auto import tqdm
from utils.io import read_json

from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

def load_model(model_name='Alibaba-NLP/gte-multilingual-reranker-base', max_length=512):
    model = CrossEncoder(model_name=model_name, trust_remote_code=True, max_length=max_length)
    model.model.config.max_position_embeddings = max_length
    model.model.to('cuda')
    return model

def pipeline(model_name, corpus, predicts, output_k=10, max_length=512):
    batched_queries = []
    batched_relevant_contexts = []
    batched_predict_ids = []
    batched_relevant_ids = []
    batched_relevant_scores = []
    
    def write_to_txt(fo, i):
        predict_id = batched_predict_ids[i * 20]
        relevant = batched_relevant_ids[i * 20:(i + 1) * 20]
        score_slice = scores[i * 20:(i + 1) * 20]
        sorted_results = sorted(zip(score_slice, relevant), key=lambda x: -x[0])[:output_k]
        if ((batched_relevant_scores[i * 20] - batched_relevant_scores[i * 20 + 1] >= embed_threshold)
                and ((batched_relevant_scores[i * 20] - batched_relevant_scores[i * 20 + 1]) /
                    (batched_relevant_scores[i * 20 + 1] - batched_relevant_scores[i * 20 + 2]) >= embed_threshold_ratio)):
            sorted_results = zip(score_slice[:output_k], relevant[:output_k])
        fo.write(str(predict_id) + ' ' + ' '.join([str(x[1]) for x in sorted_results]) + '\n')

    model = load_model(model_name, max_length=max_length)
    pbar = tqdm(enumerate(predicts), total=len(predicts))
    batch_size = config['batch_size']
    embed_threshold = config['embed_threshold']
    embed_threshold_ratio = config['embed_threshold_ratio']
    
    with open(config['output_predict_txt'], 'w') as fo:
        for k, predict in pbar:
            query = public_test[predict['id']]
            relevant_contexts = [corpus[context] for context in predict['relevant']]
            batched_queries.extend([query] * len(relevant_contexts))
            batched_relevant_contexts.extend(relevant_contexts)
            batched_predict_ids.extend([predict['id']] * len(relevant_contexts))
            batched_relevant_ids.extend(predict['relevant'])
            batched_relevant_scores.extend(predict['score'])

            if len(batched_queries) == batch_size * len(relevant_contexts):
                scores = model.predict(list(zip(batched_queries, batched_relevant_contexts)), batch_size=batch_size)
                assert len(scores) == batch_size * 20
                for i in range(batch_size):
                    write_to_txt(fo, i)

                # Clear batched lists after processing
                batched_queries.clear()
                batched_relevant_contexts.clear()
                batched_predict_ids.clear()
                batched_relevant_ids.clear()
                batched_relevant_scores.clear()

        # Process any remaining queries
        if batched_queries:
            scores = model.predict(list(zip(batched_queries, batched_relevant_contexts)))
            for i in range(len(batched_queries) // 20):
                write_to_txt(fo, i)
        return os.path.abspath(config['output_predict_txt']), mrr_score

if __name__ == '__main__':
    config = read_json(path="configs/infer_cross.json")
    corpus = read_json(config['corpus_path'])
    predicts = read_json(path=config['output_detailed_predict_json'])
    output_k = config['top_k']
    corpus = {doc['id']: doc['text'] for doc in corpus}
    public_test = {test['id']: test['text'] for test in predicts}

    results, mrr_score = pipeline(
        model_name=config['model_name'],
        corpus=corpus,
        predicts=predicts,
        output_k=config['top_k'],
        max_length=config['max_length']
    )