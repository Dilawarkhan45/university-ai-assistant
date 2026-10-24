# University AI Assistant

A local web app that answers students' questions using university PDF documents
(admission policies, exam rules, handbooks, scholarship notices). It retrieves the
most relevant passages from the PDFs you provide and asks an AI model to answer
from those passages only. Every answer names the file and page it came from.

Built step by step as a learning project with Python, PyMuPDF, Sentence
Transformers, FAISS, the Google Gemini API and Streamlit.

## Features

- Upload several PDFs, or copy them into a folder, and build a searchable index.
- Answers use only the uploaded documents and cite the file name and page.
- Cited files and pages are checked against the passages actually sent to the AI.
- Says "the documents do not provide enough information" instead of guessing.
- Shows the supporting passages in expandable sections.
- Handles empty, corrupted, password-protected and scanned PDFs without crashing.
- The index rebuilds automatically when documents or settings change.
- Passages are treated as untrusted text, which reduces the risk of instructions
  hidden inside a PDF being followed.

## How it works

1. **Extract:** PyMuPDF reads each PDF page by page (`src/pdf_processor.py`).
2. **Chunk:** each page is split into overlapping pieces of about 800 characters
   with 150 characters of overlap (`src/text_chunker.py`).
3. **Embed:** Sentence Transformers (`all-MiniLM-L6-v2`) turns each chunk into 384
   numbers (`src/embeddings.py`).
4. **Index:** FAISS stores the vectors on disk (`src/vector_store.py`,
   `src/retriever.py`).
5. **Retrieve:** a question is embedded and the closest chunks above a minimum
   similarity score are kept.
6. **Answer:** the question and the top passages go to Gemini
   (`src/llm_client.py`), which answers from them only.
7. **Check:** the answer's citations are compared with the passages sent
   (`src/rag_pipeline.py`) and shown in the web page (`app.py`).

## Requirements

- Windows 10, VS Code and PowerShell (other systems are untested).
- Python 3.13 (developed and tested on 3.13.7).
- A Google Gemini API key (free tier available, see Privacy below).
- Internet access for the first install, the first model download, and every
  question (the AI step uses the internet).

## Setup

Run these in PowerShell from the project folder.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Create your settings file and add your key:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set `GEMINI_API_KEY` to your own key. Get one at
https://aistudio.google.com/apikey. Never commit `.env` or share your key.

| Setting                 | Meaning                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`        | Your Gemini API key.                                                                                                      |
| `GEMINI_MODEL`          | The model name. Google retires models over time, so check Google's deprecation page if you get a "model not found" error. |
| `GEMINI_THINKING_LEVEL` | `minimal` keeps answers fast. Not every model supports every level.                                                       |

The first run downloads the embedding model (about 80 MB) and stores it in your
user folder; later runs reuse it.

## Run the app

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501. The app only listens on your own computer. Stop it
with Ctrl+C. Streamlit's file watcher is turned off in `.streamlit/config.toml`,
so restart the app after editing code.

## Adding documents

- **In the web page:** use "Add documents" in the sidebar, choose PDFs (up to
  25 MB each) and click "Save and process".
- **Or by hand:** copy PDFs into `data/pdfs` and click "Check for document
  changes" in the sidebar.
- "Rebuild index" forces a full rebuild. Any change to a document, or to the
  model or chunk settings, rebuilds the whole index.
- Files that cannot be used are listed in the sidebar with the reason.

To try the app without real documents, create fictional samples:

```powershell
python tests\make_sample_pdfs.py
```

## Command-line use

```powershell
python -m src.rag_pipeline --question "What is the attendance requirement?"
python -m src.retriever --rebuild
```

## Tests

```powershell
python -m src.rag_pipeline --selftest
python -m src.upload_manager --selftest
```

If you installed pytest (`pip install pytest`), run `python -m pytest`.

## Project structure

```
app.py                  Streamlit web interface
src/
  pdf_processor.py      read PDFs page by page
  text_chunker.py       split pages into overlapping chunks
  embeddings.py         text to vectors (Sentence Transformers)
  vector_store.py       FAISS index: add, search, save, load
  retriever.py          build or reuse the index, relevance check
  llm_client.py         prompt building and the Gemini client
  rag_pipeline.py       one ask() function that connects everything
  upload_manager.py     safe saving of uploaded PDFs
data/pdfs/              your PDF documents (not committed)
storage/                generated index files (not committed)
tests/                  sample generators, experiments and self-tests
.streamlit/config.toml  server settings
.env                    your secrets (not committed)
.env.example            template for .env
requirements.txt        direct dependencies, exact versions
requirements-lock.txt   full snapshot of the working environment
```

## Privacy

- PDFs and the search index stay on your computer.
- Each question sends the question text and the top retrieved passages to
  Google's Gemini service.
- On Gemini's free tier, Google's terms allow it to use submitted content to
  improve its products, and human reviewers may read it. Use only public
  documents on the free tier. For confidential material, use a paid tier with
  no-training terms (check Google's current terms) or a locally run model.
- Do not upload documents containing personal data you are not allowed to share.
- The app has no login. Anyone who can open the page can read the passages and
  use your API quota, so keep it on localhost.

## Limitations

- Scanned PDFs (pictures of pages) have no text and are skipped. OCR is not
  included.
- The embedding model is English-focused. Urdu or mixed-language documents need
  a multilingual model.
- Tables, columns and images are flattened to plain text.
- Page numbers are PDF page positions, which can differ from numbers printed on
  the page.
- The similarity cut-off (0.25) was set on small sample files. Tune it on your
  real documents. Scores are not probabilities.
- The AI can still be wrong. Citations are checked for file and page only, not
  for whether the sentence truly supports the claim. Read the passages.
- The free tier has rate limits and can be busy; the app retries and then shows
  a friendly message.
- Designed for one local user. It has no accounts or per-document permissions.

## Deployment note

Do not put this on the public internet as it is: there is no login, the free
tier may use submitted content, and PyMuPDF is distributed under the AGPL
licence (with a commercial option), so check the licence terms before offering
the app to others.

## Possible future work

OCR for scanned PDFs, Urdu support, conversation history, login and
document-level permissions, deleting documents from the web page, and a
stronger answer-checking step.
