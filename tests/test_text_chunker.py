import pytest

from src.pdf_processor import PageRecord, load_all_pdfs
from src.text_chunker import chunk_results, clean_text, create_chunks, split_text

WORDS = " ".join(f"w{i}" for i in range(300))   # w0 w1 ... w299


def word_numbers(chunk):
    return [int(word[1:]) for word in chunk.split()]


def test_short_text_stays_one_chunk():
    assert split_text("Short text.", 800, 150) == ["Short text."]


def test_empty_and_blank_text_give_no_chunks():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_chunks_never_exceed_the_size_limit():
    assert all(len(c) <= 200 for c in split_text(WORDS, 200, 40))


def test_chunks_overlap_and_always_move_forward():
    chunks = split_text(WORDS, 200, 40)
    assert len(chunks) > 3
    for before, after in zip(chunks, chunks[1:]):
        assert word_numbers(after)[0] <= word_numbers(before)[-1]   # they overlap
        assert word_numbers(after)[0] > word_numbers(before)[0]     # and make progress


def test_no_text_is_lost():
    seen = set()
    for chunk in split_text(WORDS, 200, 40):
        seen.update(word_numbers(chunk))
    assert seen == set(range(300))


def test_words_are_not_cut_in_half():
    for chunk in split_text(WORDS, 200, 40):
        assert all(word.startswith("w") and word[1:].isdigit() for word in chunk.split())


def test_chunks_end_at_sentence_boundaries_when_possible():
    text = "Sentence number one is here. " * 30
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 1
    assert all(chunk.endswith(".") for chunk in chunks)


@pytest.mark.parametrize("size, overlap", [(10, 0), (200, 100), (200, 150), (200, -1)])
def test_invalid_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text " * 50, size, overlap)


def test_clean_text_collapses_whitespace():
    assert clean_text("a   b\t\tc\r\n\r\n\r\n\r\nd") == "a b c\n\nd"


def test_create_chunks_keeps_metadata_and_unique_ids():
    pages = [
        PageRecord("a.pdf", 1, WORDS),
        PageRecord("a.pdf", 2, WORDS),
        PageRecord("b.pdf", 1, "Short page."),
    ]
    chunks = create_chunks(pages, 200, 40)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.chunk_id == f"{chunk.filename}::p{chunk.page_number}::c{chunk.chunk_index}"
    assert {(c.filename, c.page_number) for c in chunks} == {
        ("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1)}
    first_page = [c.chunk_index for c in chunks if (c.filename, c.page_number) == ("a.pdf", 1)]
    assert first_page == list(range(len(first_page)))


def test_chunk_results_skips_unusable_files_and_keeps_attribution(pdf_folder):
    chunks = chunk_results(load_all_pdfs(pdf_folder))
    assert {c.filename for c in chunks} == {
        "sample_admission_policy.pdf", "sample_exam_rules.pdf"}
    deadline = [c for c in chunks if "15 August" in " ".join(c.text.split())]
    assert [(c.filename, c.page_number) for c in deadline] == [
        ("sample_admission_policy.pdf", 2)]