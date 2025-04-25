# Note: This implementation requires the sentence-transformers library.
# Install it via: pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Load pretrained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example intent sentences
on_examples = [
    "switch on the light",
    "turn on the lamp",
    "light up the room",
    "activate the light",
    "brighten the room"
]

off_examples = [
    "switch off the light",
    "turn off the lamp",
    "darken the room",
    "deactivate the light",
    "dim the lights"
]

# Compute embeddings for example sentences
on_embeddings = model.encode(on_examples)
off_embeddings = model.encode(off_examples)

def detect_intent(prompt):
    """
    Detect intent to switch light on or off based on semantic similarity.
    Returns 'on', 'off', or 'unknown'.
    """
    prompt_embedding = model.encode([prompt])[0]

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    on_sim = max(cosine_similarity(prompt_embedding, emb) for emb in on_embeddings)
    off_sim = max(cosine_similarity(prompt_embedding, emb) for emb in off_embeddings)

    threshold = 0.5  # similarity threshold to consider intent detected
    if on_sim > off_sim and on_sim > threshold:
        return "on"
    elif off_sim > on_sim and off_sim > threshold:
        return "off"
    else:
        return "unknown"
