from config import SYSTEM_PROMPT


####### Prompt Builder #######

def build_prompt(context, history):
    """
    Builds the full prompt for the chatbot.

    The prompt has four clearly separated parts:
    1. system instructions
    2. optional retrieved context
    3. conversation history
    4. assistant marker for generation
    """

    prompt = ""

    # 1. System prompt: global behavior of the assistant
    prompt += "<|system|>\n"
    prompt += SYSTEM_PROMPT.strip() + "\n\n"

    # 2. Context block: empty for now, later used for RAG
    prompt += "<|context|>\n"

    if context.strip():
        prompt += context.strip() + "\n\n"
    else:
        prompt += "None\n\n"

    # 3. Conversation history
    for message in history:
        role = message["role"]
        content = message["content"]

        if role == "user":
            prompt += "<|user|>\n"
            prompt += content.strip() + "\n\n"

        elif role == "assistant":
            prompt += "<|assistant|>\n"
            prompt += content.strip() + "\n\n"

    # 4. Assistant marker: tells the model to generate the next assistant response
    prompt += "<|assistant|>\n"

    return prompt