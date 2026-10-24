"""Embedding generation using Sentence Transformers.

Turns text (chunks and student questions) into normalized float32 vectors.
The same model must be used for chunks and questions.
"""

import argparse
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from src.pdf_processor import load_all_pdfs
from src.text_chunker import chunk_results

# Change this one line (or pass model_name) to try another model later.
# If you change it, all chunks must be embedded again.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32

_loaded_models = {}  # keeps loaded models in memory so we load each one once


def get_model(model_name=DEFAULT_MODEL_NAME):
    """Load the embedding model (first call) or reuse it (later calls)."""
    if model_name not in _loaded_models:
        _loaded_models[model_name] = SentenceTransformer(model_name)
    return _loaded_models[model_name]


def get_embedding_dimension(model_name=DEFAULT_MODEL_NAME):
    """How many numbers the model produces per text (384 for the default)."""
    model = get_model(model_name)
    vector = model.encode(["dimension check"], normalize_embeddings=True)
    return int(vector.shape[1])


def embed_texts(texts, model_name=DEFAULT_MODEL_NAME,
                batch_size=DEFAULT_BATCH_SIZE, show_progress=False):
    """Embed a list of strings.

    Returns a float32 NumPy array of shape (number_of_texts, dimension),
    with every vector normalized to length 1.
    """
    if isinstance(texts, str):
        raise TypeError("embed_texts expects a list of strings, not one string.")
    texts = list(texts)
    if not texts:
        return np.zeros((0, get_embedding_dimension(model_name)), dtype="float32")

    model = get_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def embed_chunks(chunks, model_name=DEFAULT_MODEL_NAME,
                 batch_size=DEFAULT_BATCH_SIZE, show_progress=False):
    """Embed a list of Chunk objects. Row i of the result belongs to chunks[i]."""
    return embed_texts([chunk.text for chunk in chunks],
                       model_name, batch_size, show_progress)


def embed_query(question, model_name=DEFAULT_MODEL_NAME):
    """Embed one student question. Returns an array of shape (1, dimension)."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The question is empty.")
    return embed_texts([question.strip()], model_name)


def demo_search(chunks, embeddings, question, model_name, top_k=3):
    """Temporary demo: rank chunks by similarity (FAISS replaces this in Step 8)."""
    query = embed_query(question, model_name)[0]   # shape (dimension,)
    scores = embeddings @ query                    # one score per chunk
    best = np.argsort(-scores)[:top_k]             # indices, best score first

    print(f"\nQuestion: {question}")
    for rank, index in enumerate(best, start=1):
        chunk = chunks[index]
        print(f"  {rank}. score {scores[index]:.3f} | "
              f"{chunk.filename}, page {chunk.page_number}")
        print(f"     {' '.join(chunk.text.split())}")


def main():
    parser = argparse.ArgumentParser(description="Embed all chunks and try a search.")
    parser.add_argument("--question", help="ask your own question")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help="embedding model name")
    args = parser.parse_args()

    chunks = chunk_results(load_all_pdfs())
    if not chunks:
        print("No chunks found. Add PDFs to data/pdfs first.")
        return

    print(f"Model: {args.model}")
    start = time.time()
    embeddings = embed_chunks(chunks, args.model)
    print(f"Embedded {len(chunks)} chunks in {time.time() - start:.1f} seconds "
          "(includes loading the model)")
    print(f"Array shape: {embeddings.shape} | type: {embeddings.dtype}")
    lengths = np.linalg.norm(embeddings, axis=1)
    print(f"Vector lengths: min {lengths.min():.3f}, max {lengths.max():.3f}")

    if args.question:
        questions = [args.question]
    else:
        questions = [
            "How much attendance do I need to sit the final exam?",
            "When is the application deadline?",
            "What is the capital of France?",   # not in any document
        ]
    for question in questions:
        demo_search(chunks, embeddings, question, args.model)


if __name__ == "__main__":
    main()