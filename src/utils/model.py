from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from tokenizers import Tokenizer

def distil_model(model: str = "BAAI/bge-base-en-v1.5", vocabulary: list[str] = None, pca_dims: int = 256, apply_zipf: bool = True, device: str = "cuda") -> SentenceTransformer:
    static_embedding = StaticEmbedding.from_distillation(
                                        model_name = model,
                                        vocabulary=vocabulary,
                                        pca_dims=pca_dims,
                                        apply_zipf=apply_zipf,
                                        device=device
                                    )
    model = SentenceTransformer(modules=[static_embedding])
    return model

def distil_model_and_push_to_hub(
        model: str = "BAAI/bge-base-en-v1.5",
        vocabulary: list[str] = None,
        pca_dims: int = 256,
        apply_zipf: bool = True,
        device: str = "cuda",
        hub_id: str = 'Turbo-AI/model',
        private: bool = True
    ):
    model = distil_model(
                        model = model,
                        vocabulary=vocabulary,
                        pca_dims=pca_dims,
                        apply_zipf=apply_zipf,
                        device=device
                    )
    if 'm2v' not in hub_id:
        hub_id = hub_id + "-m2v"
    model.push_to_hub(hub_id, private=private)

def load_model2vec(model_name: str):
    static_embedding = StaticEmbedding.from_model2vec(model_name)
    model = SentenceTransformer(modules=[static_embedding])
    return model

