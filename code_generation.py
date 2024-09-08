def generate_code(model, tokenizer, prompt, conversation_manager, max_length=200):
    # Add user prompt to conversation history
    conversation_manager.add_message("user", f"Generate code: {prompt}")

    # Prepare input for the model
    input_ids = tokenizer.encode(f"Generate Python code for: {prompt}\n