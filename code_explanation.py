def explain_code(model, tokenizer, code, conversation_manager, max_length=300):
    # Add user prompt to conversation history
    conversation_manager.add_message("user", f"Explain this code:\n