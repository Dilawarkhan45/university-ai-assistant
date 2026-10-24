import pytest

from src.pdf_processor import extract_pdf
from src.upload_manager import (
    MAX_FILES_PER_UPLOAD,
    check_pdf_bytes,
    run_selftest,
    sanitize_filename,
    save_uploads,
)


class FakeUpload:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


GOOD_PDF = b"%PDF-1.4\nsome content"


def test_built_in_selftest_passes(capsys):
    run_selftest()
    assert ", 0 failed" in capsys.readouterr().out


@pytest.mark.parametrize("raw, expected", [
    ("report.pdf", "report.pdf"),
    ("../../evil.pdf", "evil.pdf"),
    ("C:\\Users\\me\\handbook.PDF", "handbook.pdf"),
    ("notes.txt", None),
    ("report.pdf.exe", None),
    ("stream.pdf:hidden", None),
    ("CON.pdf", "_CON.pdf"),
    ("", None),
])
def test_file_names_are_cleaned(raw, expected):
    assert sanitize_filename(raw) == expected


def test_content_checks():
    assert check_pdf_bytes(b"") is not None
    assert check_pdf_bytes(b"plain text") is not None
    assert check_pdf_bytes(GOOD_PDF, max_bytes=5) is not None
    assert check_pdf_bytes(GOOD_PDF) is None


def test_uploads_never_land_outside_the_folder(tmp_path):
    folder = tmp_path / "pdfs"
    result = save_uploads([FakeUpload("../../escape.pdf", GOOD_PDF)], folder)[0]
    assert result.saved_name == "escape.pdf"
    assert (folder / "escape.pdf").exists()
    assert not (tmp_path / "escape.pdf").exists()


def test_too_many_files_in_one_upload(tmp_path):
    uploads = [FakeUpload(f"file{i}.pdf", GOOD_PDF + bytes([i])) for i in range(MAX_FILES_PER_UPLOAD + 3)]
    statuses = [r.status for r in save_uploads(uploads, tmp_path / "pdfs")]
    assert statuses[:MAX_FILES_PER_UPLOAD] == ["added"] * MAX_FILES_PER_UPLOAD
    assert statuses[MAX_FILES_PER_UPLOAD:] == ["rejected"] * 3


def test_an_uploaded_pdf_can_be_read_afterwards(tmp_path, make_pdf):
    source = make_pdf(tmp_path / "source.pdf", ["Hostel fee is PKR 45,000 per semester."])
    folder = tmp_path / "pdfs"
    result = save_uploads([FakeUpload("Hostel Rules.pdf", source.read_bytes())], folder)[0]
    assert result.status == "added"
    extracted = extract_pdf(folder / result.saved_name)
    assert extracted.status == "ok"
    assert "45,000" in extracted.pages[0].text