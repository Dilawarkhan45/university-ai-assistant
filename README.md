# University AI Assistant

A local web app that answers questions using the PDF files you upload (admission
policies, exam rules, handbooks, notices). It finds the most relevant passages and
asks an AI model to answer from them. Answers are expected to cite the file and page;
check the displayed passages because the model can make mistakes.

Built step by step with Python, PyMuPDF, Sentence Transformers, FAISS, the Google
Gemini API and Streamlit.

## What it does

- Searches only the PDFs you upload in your own browser session.
- The model is instructed to answer from those documents and cite the file name and page.
- Cited files and pages are checked against the passages actually sent to the AI.
- If no relevant passages are found, says "the documents do not provide enough
  information" without sending the question to the AI.
- Shows the supporting passages in expandable sections.
- Rejects empty uploads and reports unreadable or password-protected PDFs instead
  of indexing them.
- PDFs with no extractable text are skipped; this can include scanned PDFs. There
  is no OCR support.
- Treats document text as untrusted, which reduces the risk of hidden instructions
  inside a PDF being followed.

## How it works

1. PyMuPDF extracts text page by page (`src/pdf_processor.py`).
2. Extracted page text is split into overlapping chunks of about 800 characters
   (`src/text_chunker.py`).
3. Sentence Transformers (`all-MiniLM-L6-v2`) turns each chunk into 384 numbers (`src/embeddings.py`).
4. FAISS stores the vectors and finds the closest ones (`src/vector_store.py`, `src/retriever.py`).
5. The question and the top passages go to Gemini (`src/llm_client.py`).
6. The answer's citations are checked against the passages sent (`src/rag_pipeline.py`).
7. The web page is `app.py`. Uploads are handled safely by `src/upload_manager.py`.

## Requirements

- Windows 10, VS Code and PowerShell (other systems are untested).
- Python 3.13 (developed on 3.13.7).
- A Google Gemini API key (free tier available, see Privacy).
- Internet for installing, the first model download, and every question.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks the activation script, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Open `.env` and set `GEMINI_API_KEY` to your own key from
https://aistudio.google.com/apikey. Never commit `.env` or share your key.

| Setting                 | Meaning                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `GEMINI_API_KEY`        | Your Gemini API key.                                                                    |
| `GEMINI_MODEL`          | Model name. Google retires models over time, so change it if you get "model not found". |
| `GEMINI_THINKING_LEVEL` | `minimal` keeps answers fast. Not every model supports every level.                     |

The first run downloads the embedding model (about 80 MB) and reuses it afterwards.

## Run the app

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501. The app only listens on your own computer. Stop it with
Ctrl+C. Streamlit's file watcher is off (`.streamlit/config.toml`), so restart the
app after editing code.

## Using the app

1. In the sidebar, choose one or more PDF files (up to 25 MB each) and click
   "Save and process".
2. When the page says the files were indexed, type a question and click "Ask".
3. Open the Sources to read the passages behind the answer.

Only files uploaded in the current session are searched. Refreshing the page starts
a new, empty session.

To try the app without real documents, create fictional samples and upload them:

```powershell
python tests\make_sample_pdfs.py
python tests\make_upload_samples.py
```

## Command-line tools

These read the PDFs in `data/pdfs` (not the web uploads):

```powershell
python -m src.rag_pipeline --question "What is the attendance requirement?"
python -m src.retriever --rebuild
```

## Tests

```powershell
python -m pytest
python -m src.rag_pipeline --selftest
python -m src.upload_manager --selftest
```

## Project structure

```
app.py                  Streamlit web page
src/
  pdf_processor.py      read PDFs page by page
  text_chunker.py       split pages into overlapping chunks
  embeddings.py         text to vectors
  vector_store.py       FAISS index: add, search, save, load
  retriever.py          build or reuse the index, relevance check
  llm_client.py         prompt building and the Gemini client
  rag_pipeline.py       one ask() function that connects everything
  upload_manager.py     safe saving of uploaded PDFs
data/pdfs/              PDFs for the command-line tools (not committed)
data/uploads/           files uploaded in the web page, one folder per session (not committed)
storage/                generated indexes (not committed)
tests/                  tests, sample generators and experiments
.streamlit/config.toml  server settings
.env / .env.example     your secrets / a safe template
requirements.txt        direct dependencies, exact versions
requirements-lock.txt   full snapshot of the working Windows environment
```

## Privacy

- Uploaded files and indexes stay on your computer, in a random folder per session.
- Session folders older than 24 hours are deleted when the app starts. A refreshed
  page does not delete the old session's files at once. To delete everything now,
  stop the app and run:

```powershell
Remove-Item -Recurse -Force data\uploads, storage\uploads
```

- When relevant passages are found, the question and those passages are sent to
  Google's Gemini service. Questions with no relevant passages are answered without
  a Gemini request.
- Google's data-use terms depend on billing status and can change. Check the current
  Gemini API terms before sending documents, and do not submit sensitive content
  unless you have confirmed the applicable data protections.
- The app has no login. Anyone who can open the page can use your API quota, so keep
  it on localhost.

## Limitations

- PDFs with no extractable text are skipped. Scanned PDFs without an existing text
  layer need OCR, which is not supported.
- The embedding model is English-focused. Urdu needs a multilingual model.
- Extracted text does not preserve page layout: tables and columns may be jumbled,
  and images are not interpreted.
- Page numbers are PDF page positions and can differ from numbers printed on pages.
- The starting similarity cut-off is 0.25. Tune it using your own documents; scores
  are not probabilities.
- The AI can still be wrong. Citations are checked for file and page only, not for
  whether the sentence truly supports the claim. Read the passages.
- Gemini has rate limits and can be busy. The app retries, then shows a message.
- Built for one local user. No accounts or per-document permissions.

## Deployment note

Do not put this on the public internet as it is. There is no login, Google's data-use
terms depend on the account and billing status, and PyMuPDF is under the AGPL licence
(a commercial licence is available from Artifex). Check the applicable terms before
offering the app to others. `requirements-lock.txt` is a Windows environment snapshot;
Linux compatibility is not guaranteed.

## Possible future work

OCR for scanned PDFs, Urdu support, conversation history, login and per-document
permissions, and a stronger check that each sentence is supported by its citation.
