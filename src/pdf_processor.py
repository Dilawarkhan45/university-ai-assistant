"""PDF text extraction using PyMuPDF.

Reads PDFs page by page and keeps the filename and page number with every
piece of text, so answers can be cited later.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF_FOLDER = PROJECT_ROOT / "data" / "pdfs"


@dataclass
class PageRecord:
    """Text of one page, with where it came from."""
    filename: str
    page_number: int  # human-style: first page is 1
    text: str


@dataclass
class ExtractionResult:
    """Outcome of reading one PDF file."""
    filename: str
    status: str  # "ok", "no_text", "password_protected" or "error"
    message: str
    total_pages: int = 0
    pages: list = field(default_factory=list)  # list of PageRecord


def extract_pdf(pdf_path):
    """Read one PDF and return an ExtractionResult. Never raises for bad files."""
    pdf_path = Path(pdf_path)
    filename = pdf_path.name

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        return ExtractionResult(
            filename=filename,
            status="error",
            message=f"Could not read this PDF ({type(exc).__name__}). "
                    "It may be corrupted or empty.",
        )

    try:
        if doc.needs_pass:
            return ExtractionResult(
                filename=filename,
                status="password_protected",
                message="This PDF is password-protected and was skipped.",
            )

        total_pages = doc.page_count
        pages = []
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:  # skip pages with no text
                pages.append(PageRecord(filename, index + 1, text))
    except Exception as exc:
        return ExtractionResult(
            filename=filename,
            status="error",
            message=f"Problem while reading pages ({type(exc).__name__}).",
        )
    finally:
        doc.close()

    if not pages:
        return ExtractionResult(
            filename=filename,
            status="no_text",
            message="No extractable text found. This may be a scanned PDF "
                    "(it would need OCR).",
            total_pages=total_pages,
        )

    return ExtractionResult(
        filename=filename,
        status="ok",
        message=f"Extracted text from {len(pages)} of {total_pages} page(s).",
        total_pages=total_pages,
        pages=pages,
    )


def load_all_pdfs(folder=DEFAULT_PDF_FOLDER):
    """Read every .pdf file in a folder. Returns a list of ExtractionResult."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    pdf_files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    return [extract_pdf(p) for p in pdf_files]


def preview_text(text, max_chars=200):
    """Return a short, single-line preview of some text."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def main():
    results = load_all_pdfs()
    if not results:
        print("No PDF files found in data/pdfs.")
        return

    print(f"Found {len(results)} PDF file(s) in data/pdfs\n")
    counts = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
        print(f"[{result.status.upper()}] {result.filename}")
        print(f"    {result.message}")
        if result.pages:
            first = result.pages[0]
            print(f"    Preview (page {first.page_number}): "
                  f"{preview_text(first.text)}")
        print()

    summary = ", ".join(f"{n} {status}" for status, n in counts.items())
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()