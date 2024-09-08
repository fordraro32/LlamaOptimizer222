from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(device):
    model_name = "mlabonne/Hermes-3-Llama-3.1-70B-lorablated"
    
    # Load the model with correct rope_scaling configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        rope_scaling={
            "type": "linear",
            "factor": 8.0
        }
    )
    
    # Move the model to the appropriate device
    model = model.to(device)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer
