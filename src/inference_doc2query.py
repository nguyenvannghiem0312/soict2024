import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from datasets import load_dataset
import torch

# Import functions from respective module
from ..util.io import save_to_json, read_json

parser = argparse.ArgumentParser(description="Generate queries for each item in the dataset.")
parser.add_argument("--model_name", required=True, help="The name or path of the pretrained model.")
parser.add_argument("--dataset", required=True, help="The name of the dataset to load.")
parser.add_argument("--generate_log_path", required=True, help="The file path to save the log of generated queries.")
parser.add_argument("--generate_query_path", required=True, help="The file path to save the generated queries.")
parser.add_argument("--max_length", type=int, default=512, help="The maximum length of the input text.")
parser.add_argument("--top_p", type=float, default=0.95, help="The nucleus sampling probability threshold.")
parser.add_argument("--top_k", type=int, default=10, help="The top-k sampling size.")
parser.add_argument("--num_return_sequences", type=int, default=2, help="The number of sequences to generate.")
args = parser.parse_args()

def main(args):
    """
    Generate queries for each item in the dataset using the specified model and parameters.

    Args:
    - args (argparse.Namespace): Command-line arguments.

    Returns:
    - None
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to("cuda:0")

    # Load dataset
    dataset = load_dataset(args.dataset)
    dataset = dataset['corpus']
    dataset = dataset.to_list()

    def create_queries(para, tokenizer, model):
        """
        Generate queries for a given paragraph using the specified tokenizer and model.

        Args:
        - para (str): The paragraph text.
        - tokenizer (transformers.AutoTokenizer): The tokenizer.
        - model (transformers.AutoModelForSeq2SeqLM): The model.

        Returns:
        - list: A list of generated queries.
        """
        # Encode paragraph text
        input_ids = tokenizer.encode(para, return_tensors='pt').to('cuda:0')

        with torch.no_grad():
            # Generate queries using top-k / top-p random sampling
            sampling_outputs = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                num_return_sequences=args.num_return_sequences
            )

        list_queries = []
        # Write generated queries to a log file
        with open(args.generate_log_path, "a") as log_file:
            log_file.write("Paragraph:\n")
            log_file.write(para + "\n\n")

            log_file.write("Sampling Outputs:\n")
            for i in range(len(sampling_outputs)):
                query = tokenizer.decode(sampling_outputs[i], skip_special_tokens=True)
                log_file.write(f'{i + 1}: {query}\n')
                list_queries.append(query)

        return list_queries

    # Loop through each item in the dataset and generate queries
    for item in tqdm(dataset):
        generate_query = read_json(args.generate_query_path)
        docs = item['title'] + "\n" + item['text']

        # Check if the length of the document is less than or equal to the specified max_length
        if len(docs.split(' ')) <= args.max_length:
            list_queries = create_queries(docs, tokenizer, model)
            generate_query.extend([{
                'article_id': item['article_id'],
                'query_generate': query
            } for query in list_queries])
        save_to_json(generate_query, args.generate_query_path)

if __name__ == "__main__":
    main(args)

