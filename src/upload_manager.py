"""Safe saving of uploaded PDF files into data/pdfs.

Never uses the uploaded file name as a path directly: names are cleaned, the
destination is chosen here, and files are written atomically.
"""

import argparse
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from src.pdf_processor import DEFAULT_PDF_FOLDER

MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_FILES_PER_UPLOAD = 20
MAX_STEM_LENGTH = 100
_RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


@dataclass
class UploadResult:
    original_name: str   # as sent by the browser: display only, never a path
    saved_name: str | None
    status: str          # "added", "replaced", "unchanged" or "rejected"
    message: str


def sanitize_filename(name):
    """Return a safe '.pdf' file name, or None if the name is not acceptable."""
    if not isinstance(name, str):
        return None
    base = PureWindowsPath(name.replace("\x00", "")).name   # drops folders and drives
    base = "".join(ch if (ch.isalnum() or ch in " ._-()") else "_" for ch in base)
    base = re.sub(r" {2,}", " ", base)
    if not base.lower().endswith(".pdf"):
        return None
    stem = base[:-4].strip(" .")
    if not stem:
        return None
    stem = stem[:MAX_STEM_LENGTH].strip(" .")
    if stem.split(".")[0].upper() in _RESERVED_NAMES:
        stem = "_" + stem
    return stem + ".pdf"


def check_pdf_bytes(data, max_bytes=MAX_UPLOAD_BYTES):
    """Return None if the content looks acceptable, otherwise a short reason."""
    if not data:
        return "The file is empty."
    if len(data) > max_bytes:
        return f"The file is too large (the limit is {MAX_UPLOAD_MB} MB)."
    if b"%PDF-" not in data[:1024]:
        return "This does not look like a PDF file."
    return None


def _write_atomically(target, data):
    handle, temp_name = tempfile.mkstemp(prefix=".upload-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(data)
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _reject(original, message):
    return UploadResult(original, None, "rejected", message)


def save_uploads(files, folder=DEFAULT_PDF_FOLDER, max_bytes=MAX_UPLOAD_BYTES):
    """Save uploaded files (objects with .name and .getvalue()) into a folder."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    root = folder.resolve()
    results, used_names = [], set()

    for index, upload in enumerate(files):
        original = str(getattr(upload, "name", ""))
        if index >= MAX_FILES_PER_UPLOAD:
            results.append(_reject(
                original, f"Too many files at once (the maximum is {MAX_FILES_PER_UPLOAD})."))
            continue

        safe_name = sanitize_filename(original)
        if safe_name is None:
            results.append(_reject(original, "Only .pdf files with a normal file name are accepted."))
            continue
        if safe_name.lower() in used_names:
            results.append(_reject(original, "Another file in this upload has the same name."))
            continue

        try:
            data = upload.getvalue()
        except Exception:
            results.append(_reject(original, "The file could not be read."))
            continue
        problem = check_pdf_bytes(data, max_bytes)
        if problem:
            results.append(_reject(original, problem))
            continue

        target = folder / safe_name
        if target.resolve().parent != root:   # last line of defense
            results.append(_reject(original, "The file name is not allowed."))
            continue

        try:
            existed = target.is_file()
            if existed and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(data).digest():
                status, message = "unchanged", "This file was already uploaded (same content)."
            else:
                _write_atomically(target, data)
                status = "replaced" if existed else "added"
                message = "Saved."
        except OSError:
            results.append(_reject(original, "The file could not be saved."))
            continue

        used_names.add(safe_name.lower())
        results.append(UploadResult(original, safe_name, status, message))
    return results


# ----- offline self-test -----

def run_selftest():
    class FakeUpload:
        def __init__(self, name, data):
            self.name, self._data = name, data

        def getvalue(self):
            return self._data

    counts = {"pass": 0, "fail": 0}

    def check(label, condition):
        counts["pass" if condition else "fail"] += 1
        print(f"  {'PASS' if condition else 'FAIL'}: {label}")

    name_cases = [
        ("report.pdf", "report.pdf"),
        ("../../evil.pdf", "evil.pdf"),
        ("C:\\Users\\me\\handbook.PDF", "handbook.pdf"),
        ("my file (final).pdf", "my file (final).pdf"),
        ("notes.txt", None),
        ("report.pdf.exe", None),
        ("stream.pdf:hidden", None),
        (".pdf", None),
        ("", None),
        ("CON.pdf", "_CON.pdf"),
        ("  spaced   name .pdf", "spaced name.pdf"),
        ("r\u00e9sum\u00e9_policy.pdf", "r\u00e9sum\u00e9_policy.pdf"),
        ("a<b>c|d?.pdf", "a_b_c_d_.pdf"),
        (".hidden.pdf", "hidden.pdf"),
    ]
    for raw, expected in name_cases:
        check(f"name {raw!r} -> {expected!r}", sanitize_filename(raw) == expected)
    long_name = sanitize_filename("x" * 300 + ".pdf")
    check("very long name is shortened", long_name is not None and len(long_name) == MAX_STEM_LENGTH + 4)

    good, better = b"%PDF-1.4\ncontent one", b"%PDF-1.4\ncontent two"
    with tempfile.TemporaryDirectory() as root:
        folder = Path(root) / "pdfs"
        first = save_uploads([FakeUpload("a.pdf", good)], folder)[0]
        check("new file is added", first.status == "added")
        check("same file again is unchanged",
              save_uploads([FakeUpload("a.pdf", good)], folder)[0].status == "unchanged")
        replaced = save_uploads([FakeUpload("a.pdf", better)], folder)[0]
        check("changed file is replaced", replaced.status == "replaced"
              and (folder / "a.pdf").read_bytes() == better)
        fake = save_uploads([FakeUpload("fake.pdf", b"plain text")], folder)[0]
        check("non-PDF content is rejected", fake.status == "rejected"
              and not (folder / "fake.pdf").exists())
        check("empty file is rejected",
              save_uploads([FakeUpload("e.pdf", b"")], folder)[0].status == "rejected")
        check("too large file is rejected",
              save_uploads([FakeUpload("big.pdf", good)], folder, max_bytes=10)[0].status == "rejected")
        check("non-pdf name is rejected",
              save_uploads([FakeUpload("notes.txt", good)], folder)[0].status == "rejected")
        batch = save_uploads([FakeUpload("dup.pdf", good), FakeUpload("DUP.pdf", better)], folder)
        check("duplicate name in one upload is rejected",
              [r.status for r in batch] == ["added", "rejected"])
        escape = save_uploads([FakeUpload("../escape.pdf", good)], folder)[0]
        check("path traversal stays inside the folder",
              escape.saved_name == "escape.pdf" and (folder / "escape.pdf").exists()
              and not (Path(root) / "escape.pdf").exists())
        leftovers = [p.name for p in folder.iterdir() if p.suffix == ".tmp"]
        check("no temporary files left behind", not leftovers)

    print(f"Selftest: {counts['pass']} passed, {counts['fail']} failed")


def main():
    parser = argparse.ArgumentParser(description="Upload helper checks.")
    parser.add_argument("--selftest", action="store_true", help="run the offline self-test")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    else:
        print("Use --selftest to run the checks.")


if __name__ == "__main__":
    main()