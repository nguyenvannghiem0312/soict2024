from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaModel
import torch.nn as nn

def modify_model_and_push_to_hub(model_name: str, new_max_position_embeddings: int, hub_name: str):
    # Load model and tokenizer
    model = XLMRobertaModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Modify model's config
    config = model.config
    config.max_position_embeddings = new_max_position_embeddings
    padding_idx = model.embeddings.position_embeddings.padding_idx
    
    # Clone old position embeddings
    old_position_embeddings = model.embeddings.position_embeddings.weight.data.clone()
    
    # Update model's position embeddings
    model.config = config
    model.embeddings.position_embeddings = nn.Embedding(new_max_position_embeddings, config.hidden_size, padding_idx=padding_idx)
    model.embeddings.position_embeddings.weight.data[:old_position_embeddings.size(0), :] = old_position_embeddings
    model.embeddings.position_embeddings.weight.requires_grad = True
    model.embeddings.position_embeddings.weight.data[padding_idx].zero_()
    
    # Print model summary
    print(model)
    
    # Push model and tokenizer to the hub
    model.push_to_hub(hub_name, private=True)
    tokenizer.push_to_hub(hub_name, private=True)

def main():
    model_name = 'Turbo-AI/multilingual-e5-large-instruct__trim_vocab'
    new_max_position_embeddings = 1024
    hub_name = 'Turbo-AI/multilingual-e5-large-instruct__trim_vocab-1024'
    
    # Call the function to modify and push the model
    modify_model_and_push_to_hub(model_name, new_max_position_embeddings, hub_name)

if __name__ == "__main__":
    main()