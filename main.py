import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from model_loader import load_model
from adapter_tuning import apply_adapter_tuning
from memory_optimization import optimize_memory
from conversation_manager import ConversationManager
from text_generation import generate_text
from ui import create_ui

def main():
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model and tokenizer
    model, tokenizer = load_model(device)

    # Apply adapter tuning
    model = apply_adapter_tuning(model)

    # Optimize memory usage
    model, _ = optimize_memory(model, device)

    # Initialize conversation manager
    conversation_manager = ConversationManager()

    # Create the Gradio interface
    iface = create_ui(model, tokenizer, conversation_manager)

    # Launch the interface
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
