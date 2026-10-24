"""Explore sentence embeddings. Run from the project root:

    python tests\\try_embeddings.py
"""

import time

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

print(f"Loading model '{MODEL_NAME}' (the first run downloads it)...")
start = time.time()
model = SentenceTransformer(MODEL_NAME)
print(f"Loaded in {time.time() - start:.1f} seconds")
print("Device:", model.device)
print("Max sequence length (tokens):", model.max_seq_length)

sentences = [
    "Students need 75 percent attendance to sit the final exam.",
    "You must attend at least three quarters of classes to take the final examination.",
    "The cafeteria serves lunch from noon until two o'clock.",
]

# 1. Turn sentences into embeddings
embeddings = model.encode(sentences, normalize_embeddings=True)
print("\n--- Embeddings ---")
print("Shape (sentences, dimensions):", embeddings.shape)
print("First 5 numbers of sentence 0:", np.round(embeddings[0][:5], 4))
print("Vector lengths (should be 1.0):",
      [round(float(np.linalg.norm(e)), 3) for e in embeddings])

# 2. Cosine similarity (a plain dot product, because vectors are normalized)
print("\n--- Similarity between sentences ---")
print("0 vs 1 (same meaning, different words):", round(float(embeddings[0] @ embeddings[1]), 3))
print("0 vs 2 (unrelated):                    ", round(float(embeddings[0] @ embeddings[2]), 3))

# 3. A tiny semantic search
query = "How much attendance do I need to take exams?"
query_embedding = model.encode(query, normalize_embeddings=True)
scores = embeddings @ query_embedding
print("\n--- Search ---")
print("Question:", query)
for position in np.argsort(-scores):  # best score first
    print(f"  score {scores[position]:.3f}  ->  {sentences[position]}")

# 4. Is a typical chunk under the model's token limit?
paragraph = ("Applicants must submit their transcripts, identity documents and the "
             "application fee before the deadline announced by the admissions office. ")
long_text = (paragraph * 6)[:800]  # exactly 800 characters, like a full chunk
token_count = len(model.tokenizer(long_text)["input_ids"])
print("\n--- Token check ---")
print(f"{len(long_text)} characters = {token_count} tokens "
      f"(limit: {model.max_seq_length}; count includes 2 special tokens)")