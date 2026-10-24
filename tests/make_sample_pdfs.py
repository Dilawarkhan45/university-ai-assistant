"""Creates small FICTIONAL sample PDFs in data/pdfs for testing.

Run from the project root:  python tests\\make_sample_pdfs.py
"""

from pathlib import Path

import pymupdf

PDF_FOLDER = Path(__file__).resolve().parent.parent / "data" / "pdfs"
TEXT_AREA = pymupdf.Rect(72, 72, 523, 770)


def build_pdf(filename, page_texts):
    """Create a PDF with one page per item (None = blank page)."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(TEXT_AREA, text, fontsize=11)
    doc.save(str(PDF_FOLDER / filename))
    doc.close()


def main():
    PDF_FOLDER.mkdir(parents=True, exist_ok=True)

    admission_header = "Sample University - Admission Policy (fictional sample document)\n\n"
    build_pdf("sample_admission_policy.pdf", [
        admission_header + "Page 1: Eligibility\n\n"
        "Applicants to BS programs must hold an intermediate certificate with "
        "at least 60 percent marks. Applicants who have completed their final "
        "exams but are awaiting results may apply provisionally.",

        admission_header + "Page 2: Deadlines and Fees\n\n"
        "The application deadline for the Fall semester is 15 August. The "
        "non-refundable application fee is PKR 3,000. Late applications are "
        "not accepted.",

        admission_header + "Page 3: Scholarships and Refunds\n\n"
        "The top 10 percent of admitted students receive a merit scholarship "
        "of 50 percent tuition waiver. A student who withdraws within 14 days "
        "of admission receives an 80 percent tuition fee refund.",
    ])

    exam_header = "Sample University - Examination Rules (fictional sample document)\n\n"
    build_pdf("sample_exam_rules.pdf", [
        exam_header + "Page 1: Attendance\n\n"
        "Students must have at least 75 percent attendance in a course to be "
        "allowed to sit the final examination. Students below this limit "
        "receive an F grade in the course.",

        exam_header + "Page 2: Grading and Re-takes\n\n"
        "The minimum passing grade is D, which requires 50 percent marks. A "
        "student may re-take a failed course at most twice. Mobile phones are "
        "not allowed in the examination hall.",
    ])

    # A PDF with one blank page: simulates a scanned PDF with no text.
    build_pdf("sample_no_text.pdf", [None])

    # A file that has a .pdf name but is not a real PDF.
    (PDF_FOLDER / "sample_corrupted.pdf").write_bytes(b"This is not a real PDF file.")

    # A password-protected PDF (sample password: student123).
    try:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_textbox(TEXT_AREA, "This page is protected by a password.", fontsize=11)
        doc.save(
            str(PDF_FOLDER / "sample_password_protected.pdf"),
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw="student123",
            owner_pw="owner123",
        )
        doc.close()
    except Exception as exc:
        print(f"WARNING: could not create the password-protected sample: {exc}")

    print("Sample files created in data/pdfs:")
    for path in sorted(PDF_FOLDER.glob("*.pdf")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()