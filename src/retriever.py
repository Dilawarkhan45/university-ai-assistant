"""Retriever: builds or reuses the FAISS index and finds relevant chunks.

The index is rebuilt automatically when the PDFs or the settings change.
Chunks scoring below a minimum similarity are dropped.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.embeddings import DEFAULT_MODEL_NAME, embed_chunks, embed_query
from src.pdf_processor import DEFAULT_PDF_FOLDER, load_all_pdfs
from src.text_chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_results
from src.vector_store import (
    INDEX_FILENAME,
    METADATA_FILENAME,
    STORAGE_FOLDER,
    VectorStore,
    VectorStoreError,
    index_exists,
)

MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1
DEFAULT_TOP_K = 5
# Starting value only. Scores are NOT probabilities. Retest on real PDFs.
MIN_SCORE = 0.25


@dataclass
class BuildReport:
    """What happened when the index was prepared."""
    rebuilt: bool
    reason: str
    files_found: int = 0
    files_indexed: int = 0
    chunk_count: int = 0
    skipped: list = field(default_factory=list)  # dicts: filename, status, message


@dataclass
class RetrievalResult:
    """Outcome of one question."""
    question: str
    status: str            # "ok", "no_relevant_passages" or "no_documents"
    results: list          # SearchResult objects at or above the minimum score
    best_score: float | None  # best score found before filtering


def _file_hash(path):
    """SHA-256 of a file's contents (a fingerprint of the file)."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_pdfs(folder):
    """List of {filename, size, sha256} for every .pdf file in a folder."""
    if not folder.is_dir():
        return []
    pdf_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    return [
        {"filename": p.name, "size": p.stat().st_size, "sha256": _file_hash(p)}
        for p in pdf_files
    ]


class Retriever:
    def __init__(self, pdf_folder=DEFAULT_PDF_FOLDER, storage_folder=STORAGE_FOLDER,
                 model_name=DEFAULT_MODEL_NAME, chunk_size=CHUNK_SIZE,
                 chunk_overlap=CHUNK_OVERLAP, min_score=MIN_SCORE):
        self.pdf_folder = Path(pdf_folder)
        self.storage_folder = Path(storage_folder)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_score = min_score
        self.store = None    # the loaded VectorStore, or None if no documents
        self.report = None   # BuildReport from the last load_or_build()

    # ----- change detection -----

    def _current_manifest(self):
        return {
            "format_version": MANIFEST_VERSION,
            "settings": {
                "model_name": self.model_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
            "files": fingerprint_pdfs(self.pdf_folder),
        }

    def _read_manifest(self):
        """Return the saved manifest as a dict, or None if missing/damaged."""
        path = self.storage_folder / MANIFEST_FILENAME
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _write_manifest(self, report):
        manifest = self._current_manifest()
        manifest["report"] = {
            "files_found": report.files_found,
            "files_indexed": report.files_indexed,
            "chunk_count": report.chunk_count,
            "skipped": report.skipped,
        }
        self.storage_folder.mkdir(parents=True, exist_ok=True)
        with open(self.storage_folder / MANIFEST_FILENAME, "w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

    def _check_state(self):
        """Return (is_up_to_date, reason)."""
        if not index_exists(self.storage_folder):
            return False, "no saved index found"
        saved = self._read_manifest()
        if saved is None:
            return False, "no valid manifest found"
        current = self._current_manifest()
        if saved.get("settings") != current["settings"]:
            return False, "settings changed (model, chunk size or overlap)"
        if saved.get("files") != current["files"]:
            return False, "documents changed"
        return True, "documents and settings unchanged"

    # ----- building and loading -----

    def _remove_saved_files(self):
        """Remove the generated index files so stale data is never served."""
        for name in (INDEX_FILENAME, METADATA_FILENAME, MANIFEST_FILENAME):
            (self.storage_folder / name).unlink(missing_ok=True)

    def build_index(self, reason="rebuild requested"):
        """Read all PDFs, embed their chunks, and save a fresh index."""
        results = load_all_pdfs(self.pdf_folder)
        chunks = chunk_results(results, self.chunk_size, self.chunk_overlap)
        skipped = [
            {"filename": r.filename, "status": r.status, "message": r.message}
            for r in results if r.status != "ok"
        ]
        report = BuildReport(
            rebuilt=True,
            reason=reason,
            files_found=len(results),
            files_indexed=sum(1 for r in results if r.status == "ok"),
            chunk_count=len(chunks),
            skipped=skipped,
        )

        if not chunks:
            self._remove_saved_files()
            self.store = None
            self.report = report
            return report

        embeddings = embed_chunks(chunks, self.model_name)
        store = VectorStore(embeddings.shape[1], self.model_name)
        store.add(chunks, embeddings)
        store.save(self.storage_folder)
        self._write_manifest(report)   # written last on purpose

        self.store = store
        self.report = report
        return report

    def load_or_build(self, force=False):
        """Reuse the saved index if nothing changed, otherwise rebuild it."""
        if force:
            return self.build_index("rebuild requested")

        up_to_date, reason = self._check_state()
        if not up_to_date:
            return self.build_index(reason)

        try:
            self.store = VectorStore.load(self.storage_folder)
        except VectorStoreError:
            return self.build_index("saved index was damaged")

        saved_report = (self._read_manifest() or {}).get("report", {})
        self.report = BuildReport(
            rebuilt=False,
            reason=reason,
            files_found=saved_report.get("files_found", 0),
            files_indexed=saved_report.get("files_indexed", 0),
            chunk_count=saved_report.get("chunk_count", self.store.size),
            skipped=saved_report.get("skipped", []),
        )
        return self.report

    def ensure_ready(self):
        if self.report is None:
            self.load_or_build()

    # ----- searching -----

    def retrieve(self, question, top_k=DEFAULT_TOP_K, min_score=None):
        """Find the chunks most relevant to a question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The question is empty.")

        self.ensure_ready()
        if self.store is None or self.store.size == 0:
            return RetrievalResult(question, "no_documents", [], None)

        threshold = self.min_score if min_score is None else min_score
        found = self.store.search(embed_query(question, self.model_name), top_k)
        best_score = found[0].score if found else None
        relevant = [item for item in found if item.score >= threshold]
        status = "ok" if relevant else "no_relevant_passages"
        return RetrievalResult(question, status, relevant, best_score)


def main():
    parser = argparse.ArgumentParser(description="Try the retriever.")
    parser.add_argument("--rebuild", action="store_true", help="force an index rebuild")
    parser.add_argument("--question", help="ask your own question")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    args = parser.parse_args()

    retriever = Retriever(min_score=args.min_score)
    report = retriever.load_or_build(force=args.rebuild)

    if report.rebuilt:
        print(f"Index rebuilt: {report.reason}")
    else:
        print(f"Index loaded from disk: {report.reason}")
    print(f"Files found: {report.files_found} | indexed: {report.files_indexed} "
          f"| chunks: {report.chunk_count}")
    for item in report.skipped:
        print(f"  skipped {item['filename']} [{item['status']}]")

    questions = [args.question] if args.question else [
        "How much attendance do I need to sit the final exam?",
        "When is the application deadline?",
        "What is the capital of France?",
    ]
    for question in questions:
        outcome = retriever.retrieve(question, top_k=args.top_k)
        print(f"\nQuestion: {question}")
        print(f"Status: {outcome.status} | minimum score: {args.min_score}")
        if outcome.status == "no_documents":
            print("  There are no usable documents. Add PDFs to data/pdfs.")
        elif outcome.status == "no_relevant_passages":
            print(f"  No passage reached the minimum score "
                  f"(best score: {outcome.best_score:.3f}).")
        for item in outcome.results:
            chunk = item.chunk
            print(f"  {item.rank}. score {item.score:.3f} | "
                  f"{chunk.filename}, page {chunk.page_number}")


if __name__ == "__main__":
    main()