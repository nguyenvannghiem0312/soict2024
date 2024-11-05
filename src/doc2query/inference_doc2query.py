from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import uuid
from utils.io import read_json_or_dataset, save_to_json
import argparse
import json

def load_model_and_tokenizer(model_dir):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def generate_output(model, tokenizer, contexts, config, device):
    inputs = tokenizer(contexts, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        sampling_outputs = model.generate(
            inputs["input_ids"],
            max_length=config["max_length"],
            do_sample=True,
            top_p=config["top_p"],
            top_k=config["top_k"],
            num_return_sequences=config["num_return_sequences"]
        )
    batch_size = len(contexts)
    return [[tokenizer.decode(output, skip_special_tokens=True) for output in sampling_outputs[i::batch_size]]
            for i in range(batch_size)]

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(config['model_name'])
    model.to(device)

    corpus = read_json_or_dataset(config['input_file'])
    batch_size = config["batch_size"]
    output_data = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        contexts = [item['text'] for item in batch]
        queries_batch = generate_output(model, tokenizer, contexts, config, device)

        for item, queries in zip(batch, queries_batch):
            cid = item['id']
            context = item['text']
            for query in queries:
                output = {
                    'id': str(uuid.uuid4()),
                    'text': query,
                    'relevant': [{'id': cid, 'text': context}]
                }
                output_data.append(output)

    save_to_json(data=output_data, file_path=config["output_file"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference generate query with configuration.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/configs/infer_doc2query.json", 
        help="Path to the configuration JSON file."
    )
    
    args = parser.parse_args()
    config = read_json_or_dataset(args.config_path)

    print("Config: ", json.dumps(config, indent=4, ensure_ascii=False))

    main(config=config)
