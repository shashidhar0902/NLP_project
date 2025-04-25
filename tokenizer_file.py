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
         "dim the lights"
        ]
tokenizer.train_from_iterator(texts, trainer)

def process_prompt(prompt):
    encoded = tokenizer.encode(prompt)
    tokens = tokenizer.decode(encoded.ids)
    return tokens
