"""RAG pipeline: question -> retrieval -> LLM -> answer with verified sources.

This is the only module the web interface needs to call: RAGPipeline.ask().
"""

import argparse
import re
from dataclasses import dataclass, field

from src.llm_client import (
    MAX_QUESTION_CHARS,
    NO_ANSWER_TEXT,
    LLMError,
    generate_answer,
    get_default_client,
)
from src.retriever import DEFAULT_TOP_K, Retriever
from src.text_chunker import Chunk
from src.vector_store import SearchResult, VectorStoreError


@dataclass
class Source:
    """A retrieved passage, ready to display."""
    filename: str
    page_number: int
    score: float      # similarity score, NOT a probability
    text: str
    cited: bool       # True if the answer explicitly cites this file and page


@dataclass
class RAGResponse:
    question: str
    status: str       # answered | no_answer | no_documents | llm_error | index_error | invalid_question
    answer: str
    sources: list = field(default_factory=list)       # passages to show as sources
    retrieved: list = field(default_factory=list)     # every passage that passed the score check
    citations_verified: bool = False                  # True only if citations matched the passages sent
    unverified_citations: list = field(default_factory=list)  # warning messages
    best_score: float | None = None
    model: str = ""
    seconds: float = 0.0
    error_kind: str = ""


# ----- citation checking -----

_PAGE_GROUP = r"pages?\s*(\d+(?:\s*(?:-|\u2013|\u2014|to|,|and|&)\s*\d+)*)"


def _expand_pages(group):
    """'1-3, 5' -> {1, 2, 3, 5}"""
    pages = set()
    for start, end, single in re.findall(
            r"(\d+)\s*(?:-|\u2013|\u2014|to)\s*(\d+)|(\d+)", group):
        if single:
            pages.add(int(single))
        else:
            first, last = int(start), int(end)
            if first <= last and last - first <= 100:
                pages.update(range(first, last + 1))
    return pages


def analyze_citations(answer_text, results):
    """Compare the answer's citations with the passages that were sent.

    Returns (cited, problems):
      cited    = set of (filename, page) that the answer cites AND were sent
      problems = list of messages about citations that could not be verified
    """
    sent_pages = {}
    for result in results:
        sent_pages.setdefault(result.chunk.filename, set()).add(result.chunk.page_number)
    filenames = sorted(sent_pages, key=len, reverse=True)

    cited, problems = set(), []
    for name in filenames:
        pattern = re.escape(name) + r"\s*[,;:]?\s*" + _PAGE_GROUP
        for match in re.finditer(pattern, answer_text, re.IGNORECASE):
            for page in sorted(_expand_pages(match.group(1))):
                if page in sent_pages[name]:
                    cited.add((name, page))
                else:
                    problems.append(
                        f"The answer cites {name}, page {page}, which was not among "
                        "the retrieved passages.")

    known = [name.lower() for name in filenames]
    for token in sorted(set(re.findall(r"[\w\-.]+\.pdf", answer_text, re.IGNORECASE))):
        if not any(token.lower() in name for name in known):
            problems.append(
                f"The answer mentions {token}, which is not one of the retrieved documents.")
    return cited, problems


def _to_source(result, cited):
    chunk = result.chunk
    return Source(chunk.filename, chunk.page_number, result.score, chunk.text, cited)


# ----- the pipeline -----

class RAGPipeline:
    def __init__(self, retriever=None, llm_client=None, top_k=DEFAULT_TOP_K):
        self.retriever = retriever or Retriever()
        self._llm_client = llm_client
        self.top_k = top_k

    def prepare(self, force=False):
        """Load the saved index, or build it if the PDFs or settings changed."""
        return self.retriever.load_or_build(force=force)

    def rebuild(self):
        """Rebuild the index from the PDFs in data/pdfs."""
        return self.prepare(force=True)

    def indexed_documents(self):
        """List the indexed documents with chunk and page counts."""
        self.retriever.ensure_ready()
        store = self.retriever.store
        if store is None:
            return []
        documents = {}
        for chunk in store.chunks:
            entry = documents.setdefault(
                chunk.filename, {"filename": chunk.filename, "chunks": 0, "pages": set()})
            entry["chunks"] += 1
            entry["pages"].add(chunk.page_number)
        return [
            {"filename": d["filename"], "chunks": d["chunks"],
             "pages_with_text": len(d["pages"])}
            for d in sorted(documents.values(), key=lambda d: d["filename"])
        ]

    def _get_llm_client(self):
        if self._llm_client is None:
            self._llm_client = get_default_client()   # may raise LLMError
        return self._llm_client

    def ask(self, question):
        """Answer one question. Never raises for expected problems."""
        text = " ".join(question.split()) if isinstance(question, str) else ""
        if not text:
            return RAGResponse(text, "invalid_question", "Please type a question.")
        if len(text) > MAX_QUESTION_CHARS:
            return RAGResponse(
                text[:80] + "...", "invalid_question",
                f"The question is too long (maximum {MAX_QUESTION_CHARS} characters).")

        try:
            outcome = self.retriever.retrieve(text, top_k=self.top_k)
        except (VectorStoreError, OSError):
            return RAGResponse(
                text, "index_error",
                "The document index could not be read. Please rebuild the index.")

        retrieved = [_to_source(r, False) for r in outcome.results]

        if outcome.status == "no_documents":
            return RAGResponse(
                text, "no_documents",
                "No documents are indexed yet. Add PDF files and build the index.")
        if outcome.status == "no_relevant_passages":
            return RAGResponse(text, "no_answer", NO_ANSWER_TEXT,
                               best_score=outcome.best_score)

        try:
            client = self._get_llm_client()
            answer = generate_answer(text, outcome.results, client)
        except LLMError as error:
            # Retrieval worked, so still return the passages the student can read.
            return RAGResponse(text, "llm_error", error.user_message,
                               retrieved=retrieved, best_score=outcome.best_score,
                               error_kind=error.kind)

        common = dict(best_score=outcome.best_score, model=answer.model,
                      seconds=answer.seconds)

        if answer.is_refusal:
            return RAGResponse(text, "no_answer", answer.text, retrieved=retrieved,
                               **common)

        cited, problems = analyze_citations(answer.text, answer.passages_sent)
        retrieved = [
            _to_source(r, (r.chunk.filename, r.chunk.page_number) in cited)
            for r in outcome.results
        ]
        sources = [s for s in retrieved if s.cited]
        verified = bool(sources) and not problems
        if not sources:  # nothing could be matched: show what was sent, clearly unverified
            sources = [_to_source(r, False) for r in answer.passages_sent]
            problems = problems or ["The answer does not cite a document and page."]
        return RAGResponse(text, "answered", answer.text, sources=sources,
                           retrieved=retrieved, citations_verified=verified,
                           unverified_citations=problems, **common)


# ----- self-test (no internet needed) -----

def run_selftest():
    def make(filename, page, rank):
        chunk = Chunk(f"{filename}::p{page}::c0", filename, page, 0, "text")
        return SearchResult(rank, 0.5, chunk)

    exam, admission, handbook = (
        "sample_exam_rules.pdf", "sample_admission_policy.pdf", "Student Handbook.pdf")
    results = [make(exam, 1, 1), make(exam, 2, 2), make(admission, 3, 3),
               make(handbook, 4, 4)]
    cases = [
        ("single citation", "75 percent is needed (sample_exam_rules.pdf, page 1).",
         {(exam, 1)}, 0),
        ("page range", "See sample_exam_rules.pdf, pages 1-2.",
         {(exam, 1), (exam, 2)}, 0),
        ("page list", "See (sample_exam_rules.pdf, pages 1 and 2).",
         {(exam, 1), (exam, 2)}, 0),
        ("file name with spaces", "Listed in (Student Handbook.pdf, page 4).",
         {(handbook, 4)}, 0),
        ("two files", "(sample_exam_rules.pdf, page 2; sample_admission_policy.pdf, page 3)",
         {(exam, 2), (admission, 3)}, 0),
        ("page not retrieved", "(sample_exam_rules.pdf, page 9)", set(), 1),
        ("unknown file", "(other_policy.pdf, page 1)", set(), 1),
        ("no citation", "You need 75 percent.", set(), 0),
    ]
    passed = 0
    for name, answer, expected_cited, expected_problems in cases:
        cited, problems = analyze_citations(answer, results)
        ok = cited == expected_cited and len(problems) == expected_problems
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"Selftest: {passed} passed, {len(cases) - passed} failed")


# ----- command-line demo -----

def print_response(response):
    score = f"{response.best_score:.3f}" if response.best_score is not None else "n/a"
    print(f"Status: {response.status} | best retrieval score: {score}")
    print(f"Answer: {response.answer}")
    if response.model:
        print(f"LLM: {response.model} | {response.seconds:.1f}s")
    if response.sources:
        if response.citations_verified:
            print("Sources (cited in the answer and verified):")
        else:
            print("Passages sent (citations NOT verified):")
        for source in response.sources:
            preview = " ".join(source.text.split())[:90]
            print(f"  - {source.filename}, page {source.page_number} "
                  f"(score {source.score:.3f}): {preview}...")
    for message in response.unverified_citations:
        print(f"  WARNING: {message}")
    if response.status == "llm_error" and response.retrieved:
        print("Retrieved passages (the AI service was unavailable):")
        for source in response.retrieved[:3]:
            print(f"  - {source.filename}, page {source.page_number}")


def main():
    parser = argparse.ArgumentParser(description="Try the RAG pipeline.")
    parser.add_argument("--question", help="ask your own question")
    parser.add_argument("--rebuild", action="store_true", help="force an index rebuild")
    parser.add_argument("--selftest", action="store_true",
                        help="check the citation matching (no internet needed)")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    pipeline = RAGPipeline()
    report = pipeline.prepare(force=args.rebuild)
    print(("Index rebuilt: " if report.rebuilt else "Index loaded: ") + report.reason)
    for document in pipeline.indexed_documents():
        print(f"  {document['filename']}: {document['chunks']} chunk(s), "
              f"{document['pages_with_text']} page(s) with text")

    questions = [args.question] if args.question else [
        "How much attendance do I need to sit the final exam?",
        "How many times can I re-take a failed course?",
        "What is the hostel fee?",
        "What is the capital of France?",
        "",
    ]
    for question in questions:
        print(f"\nQuestion: {question!r}")
        print_response(pipeline.ask(question))


if __name__ == "__main__":
    main()