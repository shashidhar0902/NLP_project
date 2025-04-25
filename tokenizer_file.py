from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize and train the tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
texts = ["switch on the light",
         "switch off the light",
         "turn on the lamp",
         "turn off the lamp",
         "light up the room",
         "darken the room",
         "activate the light",
         "deactivate the light",
         "brighten the room",
         "dim the lights",
         "on the light",
         "light on",
         "off the light",
        ]
tokenizer.train_from_iterator(texts, trainer)

def process_prompt(prompt):
    encoded = tokenizer.encode(prompt)
    tokens = tokenizer.decode(encoded.ids)
    return tokens

def detect_intent(prompt):
    """
    Detect intent to switch light on or off based on keywords in the prompt.
    Returns 'on', 'off', or 'unknown'.
    """
    prompt_lower = prompt.lower()
    on_keywords = ["switch on", "turn on", "light up", "activate", "brighten",
                   "on the light", "light on","lamp on"]
    off_keywords = ["switch off", "turn off", "darken", "deactivate", "dim",
                    "off the light", "lamp off",
                    "light off"]

    for phrase in on_keywords:
        if phrase in prompt_lower:
            return "on"
    for phrase in off_keywords:
        if phrase in prompt_lower:
            return "off"
    return "unknown"
