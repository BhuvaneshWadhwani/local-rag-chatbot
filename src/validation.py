from config import STOP_MARKERS


####### Output Validation #######

def validate_response(response):
    """
    Checks whether the model response is usable.
    This prevents empty, malformed, repetitive, or role-token leaking outputs.
    """

    # Check if response is empty
    if response is None or response.strip() == "":
        return "Sorry, I could not generate a valid response."

    # Remove leading/trailing whitespace
    response = response.strip()

    # Role markers should not appear in the final assistant answer
    for marker in STOP_MARKERS:
        if marker in response:
            response = response.split(marker)[0].strip()

    # Check again after removing possible leaked markers
    if response == "":
        return "Sorry, I could not generate a valid response."

    # Very simple degenerate-output check
    if len(set(response)) < 5 and len(response) > 20:
        return "Sorry, I could not generate a valid response."

    return response