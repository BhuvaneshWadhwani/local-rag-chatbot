import torch

from config import (
    device,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    REPETITION_PENALTY
)


####### Generate Response #######

def generate_response(model, tokenizer, prompt):
    """
    Generates the assistant response from the full prompt.
    """

    # Convert text prompt to token IDs
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move tensors to GPU if available
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # Remove prompt tokens
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]

    # Convert token IDs back to text
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()