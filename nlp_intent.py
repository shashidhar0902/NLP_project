# Note: This implementation requires the sentence-transformers library.
# Install it via: pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Load pretrained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example intent sentences
on_light_examples = [
    "switch on the light",
    "turn on the lamp",
    "light up the room",
    "activate the light",
    "brighten the room",
    "please turn the light on",
    "can you switch on the light",
    "lights on"
]

off_light_examples = [
    "switch off the light",
    "turn off the lamp",
    "darken the room",
    "deactivate the light",
    "dim the lights",
    "please turn the light off",
    "can you switch off the light",
    "lights off"
]

off_fan_examples = [
    "turn off the fan",
    "switch off the fan",
    "fan off",
    "deactivate the fan",
    "dim the fan",
    "please turn the fan off",
    "can you switch off the fan",
    "stop the fan"
]

on_fan_examples = [
    "turn on the fan",
    "switch on the fan",
    "fan on",
    "activate the fan",
    "brighten the fan",
    "please turn the fan on",
    "can you switch on the fan",
    "start the fan"
]

# Compute embeddings for example sentences
on_light_embeddings = model.encode(on_light_examples)
off_light_embeddings = model.encode(off_light_examples)
on_fan_embeddings = model.encode(on_fan_examples)
off_fan_embeddings = model.encode(off_fan_examples)

def detect_intent(prompt):
    """
    Detect intent to switch light or fan on or off based on semantic similarity.
    Returns one of 'on_light', 'off_light', 'on_fan', 'off_fan', or 'unknown'.
    """
    prompt_embedding = model.encode([prompt])[0]

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = {
        "on_light": max(cosine_similarity(prompt_embedding, emb) for emb in on_light_embeddings),
        "off_light": max(cosine_similarity(prompt_embedding, emb) for emb in off_light_embeddings),
        "on_fan": max(cosine_similarity(prompt_embedding, emb) for emb in on_fan_embeddings),
        "off_fan": max(cosine_similarity(prompt_embedding, emb) for emb in off_fan_embeddings)
    }

    threshold = 0.4  # lowered threshold to be more inclusive
    best_intent = max(similarities, key=similarities.get)
    if similarities[best_intent] > threshold:
        return best_intent
    else:
        return "unknown"

# For testing
#print(detect_intent("turn on the light"))
