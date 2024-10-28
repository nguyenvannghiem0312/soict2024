import argparse
import logging
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments
import sys
from datetime import datetime
import torch
import os
import json
import tqdm
from utils.io import read_json

parser = argparse.ArgumentParser()
parser.add_argument("--lang", default="vi")
parser.add_argument("--model_name", default="doc2query/msmarco-vietnamese-mt5-base-v1")
parser.add_argument("--epochs", default=4, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--max_source_length", default=1024, type=int)
parser.add_argument("--max_target_length", default=82, type=int)
parser.add_argument("--eval_size", default=1000, type=int)
args = parser.parse_args()

def load_config(config_path="configs/sbert.json"):
    config = read_json(config_path)
    return config

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def load_data():
    config = load_config(config_path="configs/sbert.json")
    with open(config["train_path"], 'r', encoding="utf-8") as f:
        train_data = json.load(f)

    train_pairs = []
    eval_pairs = []

    for item in train_data:
        for rel in item['relevant']:
            text = rel['text']
            pair = (item['text'], text)
            if len(eval_pairs) < args.eval_size:
                eval_pairs.append(pair)
            else:
                train_pairs.append(pair)

    return train_pairs, eval_pairs

def setup_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def setup_training_args(output_dir, batch_size, epochs):
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=batch_size,
        evaluation_strategy="steps",
        save_steps=1000,
        logging_steps=100,
        eval_steps=1000,
        warmup_steps=1000,
        save_total_limit=1,
        num_train_epochs=epochs,
        report_to=None,
    )

def data_collator(examples, tokenizer, max_source_length, max_target_length, fp16):
    targets = [row[0] for row in examples]
    inputs = [row[1] for row in examples]
    label_pad_token_id = -100

    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8 if fp16 else None)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=True, truncation=True, pad_to_multiple_of=8 if fp16 else None)

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else label_pad_token_id) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = torch.tensor(labels["input_ids"])
    return model_inputs

def main():
    setup_logging()
    train_pairs, eval_pairs = load_data()
    model, tokenizer = setup_model_and_tokenizer(args.model_name)

    output_dir = 'output/' + args.lang + '-' + args.model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)

    training_args = setup_training_args(output_dir, args.batch_size, args.epochs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_pairs,
        eval_dataset=eval_pairs,
        tokenizer=tokenizer,
        data_collator=lambda examples: data_collator(examples, tokenizer, args.max_source_length, args.max_target_length, training_args.fp16)
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()