"""Shared test helpers.

Tests build their own PDFs in temporary folders, so they never touch your real
data/pdfs or storage folders.
"""

from pathlib import Path

import pymupdf
import pytest

TEXT_AREA = pymupdf.Rect(72, 72, 523, 770)

ADMISSION_PAGES = [
    "Sample University - Admission Policy (fictional)\n\nPage 1: Eligibility\n\n"
    "Applicants to BS programs must hold an intermediate certificate with at "
    "least 60 percent marks.",
    "Sample University - Admission Policy (fictional)\n\nPage 2: Deadlines and Fees\n\n"
    "The application deadline for the Fall semester is 15 August. The "
    "non-refundable application fee is PKR 3,000.",
    "Sample University - Admission Policy (fictional)\n\nPage 3: Scholarships\n\n"
    "The top 10 percent of admitted students receive a merit scholarship of "
    "50 percent tuition waiver.",
]

EXAM_PAGES = [
    "Sample University - Examination Rules (fictional)\n\nPage 1: Attendance\n\n"
    "Students must have at least 75 percent attendance in a course to be "
    "allowed to sit the final examination.",
    "Sample University - Examination Rules (fictional)\n\nPage 2: Re-takes\n\n"
    "A student may re-take a failed course at most twice.",
]


def _build_pdf(path, page_texts):
    """Create a PDF with one page per item (None = a blank page)."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(TEXT_AREA, text, fontsize=11)
    doc.save(str(path))
    doc.close()
    return Path(path)


@pytest.fixture
def make_pdf():
    """Returns a function: make_pdf(path, [page texts]) -> path."""
    return _build_pdf


@pytest.fixture
def pdf_folder(tmp_path):
    """A temporary folder with 2 good PDFs and 4 problem files."""
    folder = tmp_path / "pdfs"
    folder.mkdir()
    _build_pdf(folder / "sample_admission_policy.pdf", ADMISSION_PAGES)
    _build_pdf(folder / "sample_exam_rules.pdf", EXAM_PAGES)
    _build_pdf(folder / "sample_no_text.pdf", [None])
    (folder / "sample_corrupted.pdf").write_bytes(b"This is not a real PDF file.")
    (folder / "sample_empty.pdf").write_bytes(b"")

    protected = pymupdf.open()
    protected.new_page().insert_textbox(TEXT_AREA, "Secret text.", fontsize=11)
    protected.save(
        str(folder / "sample_password_protected.pdf"),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="student123",
        owner_pw="owner123",
    )
    protected.close()
    return folder