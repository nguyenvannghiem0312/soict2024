import vocabtrimmer

trimmer = vocabtrimmer.VocabTrimmer("jinaai/jina-reranker-v2-base-multilingual", double_embedding=False)
trimmer.trim_vocab(
        path_to_save="jina-reranker-v2-base-multilingual",
        language="vi",
        dataset='Turbo-AI/data-corpus',
        dataset_column='text',
        dataset_split='train',
        target_vocab_size=None,
        min_frequency=2,
        chunk=500,
        cache_file_vocab=None,
        cache_file_frequency=None,
        overwrite=False
    )

from transformers import AutoModel, AutoTokenizer
import torch

# Load the model and tokenizer
model = AutoModel.from_pretrained('jina-reranker-v2-base-multilingual')
tokenizer = AutoTokenizer.from_pretrained('jina-reranker-v2-base-multilingual')

# Convert the model to bf16
model = model.to(torch.bfloat16)

# Define your model repository name (it should be unique)
repo_name = "Turbo-AI/jina-reranker-v2-base-multilingual__trim_vocab"

# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)