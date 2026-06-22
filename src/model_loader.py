import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import model_name, device


####### Data Information #######

def print_data_information():
    print(f"Selected device: {device}")
    print(f"torch version: {torch.__version__}")
    print(f"transformers version: {transformers.__version__}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU acceleration is not available; running on CPU.")


####### Loading Model #######

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model = model.to(device)

    # Frozen model: inference only, no fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    return tokenizer, model


def print_model_information(model):
    print(f"\nModel loaded: {model_name}")
    print(f"Running on: {device}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    print("\nProperties:")
    print(f"Max input length:  {model.config.max_position_embeddings}")
    print(f"Vocab size:        {model.config.vocab_size}")
    print(f"Hidden size:       {model.config.hidden_size}")

    print("\nArchitecture description:")
    print(f"Hidden layers:     {model.config.num_hidden_layers}")
    print(f"Attention heads:   {model.config.num_attention_heads}")
    print(f"Intermediate size: {model.config.intermediate_size}")
    print(f"Model type:        {model.config.model_type} (Causal Decoder-only)")