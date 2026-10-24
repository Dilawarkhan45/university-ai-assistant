"""Text chunking: split page text into overlapping chunks.

Each chunk keeps the filename and page number of the page it came from,
so answers can be cited later.
"""

import argparse
import re
from dataclasses import dataclass

from src.pdf_processor import load_all_pdfs, preview_text

# Default settings, measured in CHARACTERS (not tokens or words).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Places where we prefer to cut a chunk, from best to worst.
SEPARATORS = ("\n\n", ". ", "? ", "! ", ".\n", "\n", " ")


@dataclass
class Chunk:
    """One piece of text, with where it came from."""
    chunk_id: str      # unique, e.g. "policy.pdf::p3::c0"
    filename: str
    page_number: int   # human-style: first page is 1
    chunk_index: int   # position of this chunk within its page, from 0
    text: str


def clean_text(text):
    """Tidy whitespace so chunks are consistent."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)      # many spaces/tabs -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)   # 3+ line breaks -> 2
    return text.strip()


def _find_cut(text, start, end, chunk_size):
    """Pick where a chunk should end, searching backward from `end`.

    We never cut before the halfway point, so chunks are not tiny.
    """
    min_end = start + chunk_size // 2
    window = text[min_end:end]
    for separator in SEPARATORS:
        position = window.rfind(separator)  # last occurrence in the window
        if position != -1:
            return min_end + position + len(separator)
    return end  # no good place found: cut at the size limit


def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split one string into overlapping chunks. Returns a list of strings."""
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50 characters.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size // 2:
        raise ValueError(
            "chunk_overlap must be 0 or more and less than half of chunk_size."
        )

    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while True:
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _find_cut(text, start, end, chunk_size)

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= len(text):
            break

        # Next chunk starts a little BEFORE this one ended (the overlap),
        # moved forward to the beginning of a word.
        start = end - chunk_overlap
        while start < end and not text[start - 1].isspace():
            start += 1

    return chunks


def create_chunks(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Turn a list of PageRecord objects into a list of Chunk objects."""
    chunks = []
    for page in pages:
        pieces = split_text(page.text, chunk_size, chunk_overlap)
        for index, piece in enumerate(pieces):
            chunk_id = f"{page.filename}::p{page.page_number}::c{index}"
            chunks.append(
                Chunk(chunk_id, page.filename, page.page_number, index, piece)
            )
    return chunks


def chunk_results(results, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Chunk every successfully extracted PDF. Other results are skipped."""
    chunks = []
    for result in results:
        if result.status == "ok":
            chunks.extend(create_chunks(result.pages, chunk_size, chunk_overlap))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Preview text chunking.")
    parser.add_argument("--size", type=int, default=CHUNK_SIZE,
                        help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP,
                        help="overlap in characters")
    parser.add_argument("--show", type=int, default=6,
                        help="how many chunks to display")
    args = parser.parse_args()

    results = load_all_pdfs()
    skipped = [r for r in results if r.status != "ok"]
    chunks = chunk_results(results, args.size, args.overlap)

    print(f"Chunk size: {args.size}, overlap: {args.overlap}")
    print(f"Skipped files (no usable text): {len(skipped)}")
    if not chunks:
        print("No chunks created. Add PDFs to data/pdfs first.")
        return

    print(f"Total chunks: {len(chunks)}")
    per_file = {}
    for chunk in chunks:
        per_file[chunk.filename] = per_file.get(chunk.filename, 0) + 1
    for name, count in per_file.items():
        print(f"  {name}: {count} chunk(s)")

    lengths = [len(c.text) for c in chunks]
    print(f"Chunk length: min {min(lengths)}, max {max(lengths)}, "
          f"average {sum(lengths) / len(lengths):.0f}")

    print(f"\nFirst {min(args.show, len(chunks))} chunk(s):")
    for chunk in chunks[:args.show]:
        print(f"- {chunk.chunk_id} ({len(chunk.text)} chars)")
        print(f"    starts: {preview_text(chunk.text[:60])}")
        print(f"    ends:   {preview_text(chunk.text[-60:])}")


if __name__ == "__main__":
    main()