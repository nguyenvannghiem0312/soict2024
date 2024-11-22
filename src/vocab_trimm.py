import vocabtrimmer
from utils.preprocess import TextPreprocessor
from utils.io import read_json_or_dataset, save_to_json
from utils.dataset import search_by_id
from tqdm import tqdm
from bm25s.hf import BM25HF
import json
import argparse

def trim_vocab_model(config):
    trimmer = vocabtrimmer.VocabTrimmer(config["model"], double_embedding=False)
    trimmer.trim_vocab(
            path_to_save=config["path_to_save"],
            language="vi",
            dataset=config["dataset"],
            dataset_column=config["dataset_column"],
            dataset_split=config["dataset_split"],
            target_vocab_size=None,
            min_frequency=2,
            chunk=500,
            cache_file_vocab=None,
            cache_file_frequency=None,
            overwrite=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BM25 pipeline with configuration.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        default="src/configs/trimm_vocab_config.json", 
        help="Path to the configuration JSON file."
    )
    
    args = parser.parse_args()
    config = read_json_or_dataset(args.config_path)
    trim_vocab_model(config)
