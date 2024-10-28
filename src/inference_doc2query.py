import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directory of the trained model")
    parser.add_argument("--input_text", required=True, help="Input text for inference")
    parser.add_argument("--max_length", default=64, type=int, help="Maximum length of the generated output")
    return parser.parse_args()

def load_model_and_tokenizer(model_dir):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def generate_output(model, tokenizer, input_text, max_length):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parse_arguments()
    model, tokenizer = load_model_and_tokenizer(args.model_dir)
    output_text = generate_output(model, tokenizer, args.input_text, args.max_length)
    print("Generated Output:", output_text)

if __name__ == "__main__":
    main()
    # usage python inference.py --model_dir path/to/model --input_text "Your input text here"