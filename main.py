import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from model_loader import (
    print_data_information,
    load_model,
    print_model_information
)
from chat import chat_loop


####### Main #######

print_data_information()

tokenizer, model = load_model()

print_model_information(model)

chat_loop(model, tokenizer)