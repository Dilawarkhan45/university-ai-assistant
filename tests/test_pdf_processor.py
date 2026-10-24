from src.pdf_processor import extract_pdf, load_all_pdfs, preview_text


def flat(text):
    """Collapse line breaks so wrapped text can be searched."""
    return " ".join(text.split())


def test_extracts_text_page_by_page(pdf_folder):
    result = extract_pdf(pdf_folder / "sample_admission_policy.pdf")
    assert result.status == "ok"
    assert result.total_pages == 3
    assert [p.page_number for p in result.pages] == [1, 2, 3]
    assert "15 August" in flat(result.pages[1].text)


def test_filename_is_kept_on_every_page(pdf_folder):
    result = extract_pdf(pdf_folder / "sample_exam_rules.pdf")
    assert {p.filename for p in result.pages} == {"sample_exam_rules.pdf"}


def test_page_without_text_is_skipped_but_counted(tmp_path, make_pdf):
    path = make_pdf(tmp_path / "mixed.pdf", ["First page text", None, "Third page text"])
    result = extract_pdf(path)
    assert result.status == "ok"
    assert result.total_pages == 3
    assert [p.page_number for p in result.pages] == [1, 3]


def test_pdf_without_text_is_reported_as_possible_scan(pdf_folder):
    result = extract_pdf(pdf_folder / "sample_no_text.pdf")
    assert result.status == "no_text"
    assert result.pages == []
    assert "OCR" in result.message


def test_corrupted_pdf_is_reported_not_raised(pdf_folder):
    result = extract_pdf(pdf_folder / "sample_corrupted.pdf")
    assert result.status == "error"
    assert result.pages == []


def test_error_messages_do_not_contain_private_paths(pdf_folder):
    result = extract_pdf(pdf_folder / "sample_corrupted.pdf")
    assert str(pdf_folder) not in result.message


def test_empty_file_is_never_treated_as_ok(pdf_folder):
    result = extract_pdf(pdf_folder / "sample_empty.pdf")
    assert result.status != "ok"
    assert result.pages == []


def test_password_protected_pdf_is_skipped(pdf_folder):
    result = extract_pdf(pdf_folder / "sample_password_protected.pdf")
    assert result.status == "password_protected"
    assert result.pages == []


def test_missing_file_is_reported_not_raised(tmp_path):
    result = extract_pdf(tmp_path / "does_not_exist.pdf")
    assert result.status == "error"


def test_load_all_pdfs_handles_a_mixed_folder(pdf_folder):
    results = load_all_pdfs(pdf_folder)
    counts = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    assert len(results) == 6
    assert counts["ok"] == 2
    assert counts["no_text"] == 1
    assert counts["password_protected"] == 1
    assert counts["error"] + counts.get("no_text", 0) >= 3   # corrupted, empty, blank


def test_load_all_pdfs_ignores_other_file_types(pdf_folder):
    (pdf_folder / "notes.txt").write_text("not a pdf")
    names = {r.filename for r in load_all_pdfs(pdf_folder)}
    assert "notes.txt" not in names


def test_load_all_pdfs_accepts_uppercase_extension(tmp_path, make_pdf):
    make_pdf(tmp_path / "UPPER.PDF", ["Some text here"])
    results = load_all_pdfs(tmp_path)
    assert [r.status for r in results] == ["ok"]


def test_load_all_pdfs_missing_folder_gives_empty_list(tmp_path):
    assert load_all_pdfs(tmp_path / "no_such_folder") == []


def test_preview_text_collapses_whitespace_and_shortens():
    assert preview_text("a   b\n\nc") == "a b c"
    shortened = preview_text("word " * 100, max_chars=20)
    assert shortened.endswith("...") and len(shortened) <= 23