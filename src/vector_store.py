"""FAISS vector store: index chunk embeddings, search them, save and load.

FAISS stores only vectors. The chunk metadata (filename, page, text) is kept
in a separate list in the SAME ORDER, so position i in FAISS = chunks[i].
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from src.embeddings import DEFAULT_MODEL_NAME, embed_chunks, embed_query
from src.pdf_processor import PROJECT_ROOT, load_all_pdfs
from src.text_chunker import Chunk, chunk_results

STORAGE_FOLDER = PROJECT_ROOT / "storage"
INDEX_FILENAME = "faiss.index"
METADATA_FILENAME = "metadata.json"
FORMAT_VERSION = 1


class VectorStoreError(Exception):
    """Any problem with the vector store, with a message safe to show users."""


@dataclass
class SearchResult:
    rank: int        # 1 = best match
    score: float     # similarity (cosine for normalized vectors); NOT a probability
    chunk: Chunk     # has filename, page_number, text, chunk_id


def index_exists(folder=STORAGE_FOLDER):
    """True if both saved files are present."""
    folder = Path(folder)
    return (folder / INDEX_FILENAME).is_file() and (folder / METADATA_FILENAME).is_file()


class VectorStore:
    def __init__(self, dimension, model_name=DEFAULT_MODEL_NAME):
        if int(dimension) <= 0:
            raise VectorStoreError("The embedding dimension must be positive.")
        self.dimension = int(dimension)
        self.model_name = model_name
        self.index = faiss.IndexFlatIP(self.dimension)  # exact inner-product search
        self.chunks = []                                # same order as the index

    @property
    def size(self):
        """Number of chunks stored."""
        return self.index.ntotal

    def add(self, chunks, embeddings):
        """Add chunks and their embeddings (row i of embeddings = chunks[i])."""
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim != 2:
            raise VectorStoreError("Embeddings must be a 2D array.")
        if len(chunks) != embeddings.shape[0]:
            raise VectorStoreError(
                f"Got {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )
        if embeddings.shape[1] != self.dimension:
            raise VectorStoreError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"the index dimension {self.dimension}."
            )
        if len(chunks) == 0:
            return
        self.index.add(np.ascontiguousarray(embeddings))
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=5):
        """Return up to top_k SearchResult objects, best first."""
        if self.size == 0:
            return []  # an empty index has nothing to find
        if top_k < 1:
            raise VectorStoreError("top_k must be at least 1.")

        query = np.asarray(query_embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] != self.dimension:
            raise VectorStoreError(
                f"The question embedding has the wrong shape {query.shape}; "
                f"expected one vector with {self.dimension} numbers. "
                "Was a different embedding model used?"
            )

        k = min(top_k, self.size)
        scores, positions = self.index.search(np.ascontiguousarray(query), k)

        results = []
        for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
            if position == -1:   # FAISS uses -1 for "no result"
                continue
            results.append(SearchResult(rank, float(score), self.chunks[int(position)]))
        return results

    def save(self, folder=STORAGE_FOLDER):
        """Write the FAISS index and the metadata JSON to a folder."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / INDEX_FILENAME))
        metadata = {
            "format_version": FORMAT_VERSION,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "chunk_count": len(self.chunks),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        with open(folder / METADATA_FILENAME, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder=STORAGE_FOLDER):
        """Load a saved store. Raises VectorStoreError if anything is wrong."""
        folder = Path(folder)
        index_path = folder / INDEX_FILENAME
        metadata_path = folder / METADATA_FILENAME
        if not index_path.is_file() or not metadata_path.is_file():
            raise VectorStoreError("No saved index was found. Build the index first.")

        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)
            if metadata["format_version"] != FORMAT_VERSION:
                raise ValueError("unknown format version")
            chunks = [Chunk(**item) for item in metadata["chunks"]]
            dimension = int(metadata["dimension"])
            model_name = metadata["model_name"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise VectorStoreError(
                "The saved metadata is missing or damaged. Rebuild the index."
            ) from exc

        try:
            index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorStoreError(
                "The saved FAISS index could not be read. Rebuild the index."
            ) from exc

        if index.d != dimension or index.ntotal != len(chunks):
            raise VectorStoreError(
                "The saved index and metadata do not match. Rebuild the index."
            )

        store = cls(dimension, model_name)
        store.index = index
        store.chunks = chunks
        return store


def show_results(question, results):
    print(f"\nQuestion: {question}")
    if not results:
        print("  (no results)")
    for result in results:
        chunk = result.chunk
        print(f"  {result.rank}. score {result.score:.3f} | "
              f"{chunk.filename}, page {chunk.page_number}")
        print(f"     {' '.join(chunk.text.split())[:110]}...")


def main():
    chunks = chunk_results(load_all_pdfs())
    if not chunks:
        print("No chunks found. Add PDFs to data/pdfs first.")
        return

    embeddings = embed_chunks(chunks)
    store = VectorStore(dimension=embeddings.shape[1], model_name=DEFAULT_MODEL_NAME)
    store.add(chunks, embeddings)
    print(f"Indexed {store.size} chunks (dimension {store.dimension})")

    store.save()
    print("Saved index and metadata to the storage folder")

    # Reload from disk, as the app will do later: no re-embedding of chunks.
    loaded = VectorStore.load()
    print(f"Loaded {loaded.size} chunks from disk")

    questions = [
        "How much attendance do I need to sit the final exam?",
        "When is the application deadline?",
        "What is the capital of France?",
    ]
    for question in questions:
        show_results(question, loaded.search(embed_query(question), top_k=3))

    print("\n--- Safety checks ---")
    try:
        loaded.search(np.zeros((1, 10), dtype="float32"))
    except VectorStoreError as error:
        print("Wrong dimension ->", error)
    print("Empty index ->", VectorStore(loaded.dimension).search(embed_query("test")))


if __name__ == "__main__":
    main()