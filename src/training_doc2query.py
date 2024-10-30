import logging
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments
import sys
import torch
from utils.io import read_json

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

def load_data(config):
    train_data = read_json(path=config['train_path'])

    train_pairs = []
    eval_pairs = []

    for item in train_data:
        for rel in item['relevant']:
            text = rel['text']
            pair = (item['text'], text)
            if len(eval_pairs) < config["eval_size"]:
                eval_pairs.append(pair)
            else:
                train_pairs.append(pair)

    return train_pairs, eval_pairs

def setup_model_and_tokenizer(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def setup_training_args(config):
    return Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        bf16=config["bf16"],
        per_device_train_batch_size=config["batch_size"],
        evaluation_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        save_strategy=config["save_strategy"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        warmup_ratio=config["warmup_ratio"],
        save_total_limit=config["save_total_limit"],
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        run_name=config["run_name"]
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
    config = read_json(path="configs/doc2query_config.json")
    train_pairs, eval_pairs = load_data(config)
    model, tokenizer = setup_model_and_tokenizer(config["model_name"])

    training_args = setup_training_args(config)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_pairs,
        eval_dataset=eval_pairs,
        tokenizer=tokenizer,
        data_collator=lambda examples: data_collator(examples, tokenizer, config["max_source_length"], config["max_target_length"], training_args.fp16)
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()