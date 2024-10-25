import gradio as gr
from utils.io import read_json

corpus = read_json('data/Legal Document Retrieval/corpus.json')

predict = read_json('outputs/detailed_predict.json')

query_index = 0

def search_by_id(context_id):
    for context in corpus:
        if context['id'] == context_id:
            return context['text']
    return "Context not found"

def update_query(query_idx):
    query = predict[query_idx]
    query_text = query['text']
    query_id = query['id']
    
    # Lấy top 5 context và score
    top_contexts = query['relevant'][:5]
    top_scores = query['score'][:5]
    
    # Lấy text của các context từ corpus
    context_texts = [search_by_id(ctx_id) for ctx_id in top_contexts]
    
    return query_id, query_text, context_texts, top_scores


def next_query():
    global query_index
    query_index = (query_index + 1) % len(predict)
    query_id, query_text, context_texts, scores = update_query(query_index)
    return [query_id, query_text] + context_texts + scores


def previous_query():
    global query_index
    query_index = (query_index - 1) % len(predict)
    query_id, query_text, context_texts, scores = update_query(query_index)
    return [query_id, query_text] + context_texts + scores

with gr.Blocks() as demo:
    with gr.Row():
        query_id_text = gr.Textbox(label="ID Query", interactive=False)
        query_text = gr.Textbox(label="Query Text", interactive=False)
    
    with gr.Row():
        context_text_boxes = [gr.Textbox(label=f"Top Context {i+1}", interactive=False) for i in range(5)]
    
    with gr.Row():
        score_boxes = [gr.Textbox(label=f"Score {i+1}", interactive=False) for i in range(5)]
    
    with gr.Row():
        prev_button = gr.Button("Previous")
        next_button = gr.Button("Next")
    
    query_id, query_text_value, context_texts, scores = update_query(query_index)
    outputs = [query_id_text, query_text] + context_text_boxes + score_boxes
    
    # Cập nhật khi nhấn nút
    next_button.click(fn=next_query, inputs=[], outputs=outputs)
    prev_button.click(fn=previous_query, inputs=[], outputs=outputs)

demo.launch()