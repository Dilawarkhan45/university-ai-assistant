"""Creates test files for the upload feature in tests/upload_samples.

These are NOT placed in data/pdfs. You upload them through the web page.
Run from the project root:   python tests\\make_upload_samples.py
"""

from pathlib import Path

import pymupdf

FOLDER = Path(__file__).resolve().parent / "upload_samples"
TEXT_AREA = pymupdf.Rect(72, 72, 523, 770)


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        TEXT_AREA,
        "Sample University - Hostel Rules (fictional sample document)\n\n"
        "Page 1: Hostel Fees\n\n"
        "The hostel fee is PKR 45,000 per semester. Hostel residents must return "
        "to the hostel by 10 pm on weekdays.",
        fontsize=11,
    )
    doc.save(str(FOLDER / "hostel_rules.pdf"))
    doc.close()

    blank = pymupdf.open()
    blank.new_page()                       # a page with no text, like a scan
    blank.save(str(FOLDER / "blank_pages.pdf"))
    blank.close()

    (FOLDER / "not_really_a_pdf.pdf").write_bytes(b"This is plain text, not a PDF.")

    print("Created in tests/upload_samples:")
    for path in sorted(FOLDER.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()