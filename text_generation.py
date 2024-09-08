def generate_text(model, tokenizer, prompt, conversation_manager, max_length=100):
    # Add user prompt to conversation history
    conversation_manager.add_message("user", prompt)

    # Prepare input for the model
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate response
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Add model response to conversation history
    conversation_manager.add_message("assistant", response)

    return response
