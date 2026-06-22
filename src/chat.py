from config import MAX_HISTORY_MESSAGES
from prompt_builder import build_prompt
from generation import generate_response
from validation import validate_response


####### Token Monitoring #######

def count_tokens(tokenizer, text):
    tokens = tokenizer(text, add_special_tokens=False)
    return len(tokens["input_ids"])


####### Chat Loop #######

def chat_loop(model, tokenizer):
    context = ""
    conversation_history = []

    print("\nChatbot is ready!")
    print("Type 'quit' or 'exit' to stop the chat.\n")

    while True:
        user_input = input("User: ")

        if user_input.lower().strip() in ["quit", "exit"]:
            print("Goodbye!")
            break

        conversation_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        prompt = build_prompt(
            context,
            conversation_history
        )

        num_tokens = count_tokens(tokenizer, prompt)
        print(f"\nCurrent token count: {num_tokens}")

        response = generate_response(model, tokenizer, prompt)
        response = validate_response(response)

        print("\nAssistant:")
        print(response)
        print()

        conversation_history.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        # Keep only the most recent messages
        if len(conversation_history) > MAX_HISTORY_MESSAGES:
            conversation_history = conversation_history[-MAX_HISTORY_MESSAGES:]
            print("Old messages removed from history.")