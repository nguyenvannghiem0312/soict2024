import vocabtrimmer

trimmer = vocabtrimmer.VocabTrimmer("intfloat/multilingual-e5-large", double_embedding=False)
trimmer.trim_vocab(
        path_to_save="multilingual-e5-large",
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
model = AutoModel.from_pretrained('multilingual-e5-large')
tokenizer = AutoTokenizer.from_pretrained('multilingual-e5-large')

# Convert the model to bf16
model = model.to(torch.bfloat16)

# Define your model repository name (it should be unique)
repo_name = "Turbo-AI/multilingual-e5-large__trim_vocab"

# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)