import os
import sys
import json
import torch
import logging
from datetime import datetime
from torch.utils.data import Dataset, IterableDataset
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments

with open('configs/infer_sbert.json', 'r') as f:
    config = json.load(f)

with open('configs/sbert.json', 'r') as f:
    sbert_config = json.load(f)


corpus_path = config['corpus_path']
query_path = config['query_path']
train_path = sbert_config['train_path']

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Define arguments directly
args = {
    "lang": "en",  # Set your language here
    "model_name": "google/mt5-base",
    "epochs": 4,
    "batch_size": 32,
    "max_source_length": 320,
    "max_target_length": 64,
    "eval_size": 1000,
}

def main():
    ############ Load dataset
    with open(train_path, 'r', encoding="utf-8") as f:
        train_data = json.load(f)

    train_pairs = []
    eval_pairs = []

    for item in train_data:
        for rel in item['relevant']:
            text = rel['text']
            pair = (item['text'], text)
            if len(eval_pairs) < args["eval_size"]:
                eval_pairs.append(pair)
            else:
                train_pairs.append(pair)

    print(f"Train pairs: {len(train_pairs)}")

    ############ Model
    model = AutoModelForSeq2SeqLM.from_pretrained(args["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])

    save_steps = 1000

    output_dir = 'output/'+args["lang"]+'-'+args["model_name"].replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Output dir:", output_dir)

    # Write self to path
    os.makedirs(output_dir, exist_ok=True)

    train_script_path = os.path.join(output_dir, 'train_script.py')
    with open(train_script_path, 'w') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    ####

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=args["batch_size"],
        evaluation_strategy="steps",
        save_steps=save_steps,
        logging_steps=100,
        eval_steps=save_steps, #logging_steps,
        warmup_steps=1000,
        save_total_limit=1,
        num_train_epochs=args["epochs"],
    )

    ############ Arguments

    ############ Load datasets

    print("Input:", train_pairs[0][1])
    print("Target:", train_pairs[0][0])

    print("Input:", eval_pairs[0][1])
    print("Target:", eval_pairs[0][0])

    def data_collator(examples):
        targets = [row[0] for row in examples]
        inputs = [row[1] for row in examples]
        label_pad_token_id = -100

        model_inputs = tokenizer(inputs, max_length=args["max_source_length"], padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8 if training_args.fp16 else None)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args["max_target_length"], padding=True, truncation=True, pad_to_multiple_of=8 if training_args.fp16 else None)

        # replace all tokenizer.pad_token_id in the labels by -100 to ignore padding in the loss.
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else label_pad_token_id) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = torch.tensor(labels["input_ids"])
        return model_inputs

    ## Define the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_pairs,
        eval_dataset=eval_pairs,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    ### Save the model
    train_result = trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()