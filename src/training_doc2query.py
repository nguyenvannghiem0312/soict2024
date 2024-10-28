import argparse
import logging
import os
import sys
from datetime import datetime
from shutil import copyfile
import torch
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import Dataset, IterableDataset
from transformers import Seq2SeqTrainer, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments
import tqdm

# Import constants and functions from respective modules
from ..util.constant import CORPUS_LEGAL_ZALO, TRAIN_ZALO
from ..util.io import read_json
from ..util.datasets import read_corpus

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lang", required=True, help="Language of the text (e.g., Vietnamese, English)")
parser.add_argument("--model_name", default="doc2query/msmarco-vietnamese-mt5-base-v1", help="Name of the used model")
parser.add_argument("--epochs", default=4, type=int, help="Number of training epochs")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--max_source_length", default=1024, type=int, help="Maximum length of the source input")
parser.add_argument("--max_target_length", default=82, type=int, help="Maximum length of the target output")
parser.add_argument("--eval_size", default=500, type=int, help="Size of the evaluation dataset")
parser.add_argument("--bf", default=True, type=bool, help="Whether to use mixed precision training with bfloat16")
parser.add_argument("--evaluation_strategy", default="steps", choices=["steps", "epoch"], help="The evaluation strategy to use")
parser.add_argument("--logging_steps", default=100, type=int, help="Log training loss every given number of steps")
parser.add_argument("--save_steps", default=1000, type=int, help="Save model checkpoint every given number of steps")
parser.add_argument("--warmup_steps", default=1000, type=int, help="Number of warmup steps")
parser.add_argument("--save_total_limit", default=1, type=int, help="Limit the total number of checkpoints to save")
parser.add_argument("--report_to", default="wandb", help="The platform to report training metrics to or set none")

args = parser.parse_args()

def main():
    ############ Load dataset

    # Load corpus and queries
    corpus = read_corpus(CORPUS_LEGAL_ZALO)
    queries = read_json(TRAIN_ZALO)

    train_pairs = []
    eval_pairs = []

    # Generate training and evaluation pairs
    for query in queries:
        text_query = query['question']
        id_doc = query['relevant_articles']

        for did in id_doc:
            text_doc = corpus[did]

            if len(eval_pairs) < args.eval_size:
                eval_pairs.append({
                    "query": text_query,
                    "doc": text_doc
                })
            else:
                train_pairs.append({
                    "query": text_query,
                    "doc": text_doc
                })

    data = DatasetDict({
        "train": Dataset.from_list(train_pairs),
        "eval": Dataset.from_list(eval_pairs)
    })
    print(f"Train pairs: {len(train_pairs)}")

    # data.push_to_hub('NghiemAbe/doc2query') # Needs authentication and access rights

    ############ Model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    output_dir = f'output/{args.lang}-{args.model_name.replace("/", "-")}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    print("Output dir:", output_dir)

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Copy the train script to the output directory
    train_script_path = os.path.join(output_dir, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    ####

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        bf16=args.bf,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy=args.evaluation_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.epochs,
        report_to=args.report_to,  # You need to provide authentication and project info for wandb
    )

    ############ Arguments

    print("Input:", train_pairs[0]['doc'])
    print("Target:", train_pairs[0]['query'])

    print("Input:", eval_pairs[0]['doc'])
    print("Target:", eval_pairs[0]['query'])

    def data_collator(examples):
        targets = [row['query'] for row in examples]
        inputs = [row['doc'] for row in examples]
        label_pad_token_id = -100

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=8 if training_args.fp16 else None)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_length, padding=True, truncation=True, pad_to_multiple_of=8 if training_args.fp16 else None)

        # Replace all tokenizer.pad_token_id in the labels by -100 to ignore padding in the loss.
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
