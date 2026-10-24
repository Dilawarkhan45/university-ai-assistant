"""University AI Assistant - Streamlit interface.

Only the PDF files a person uploads in their own session are searched.
Run from the project root:   python -m streamlit run app.py
"""

import re
import shutil
import time
import traceback
import uuid

import streamlit as st

from src.pdf_processor import PROJECT_ROOT
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.upload_manager import MAX_UPLOAD_MB, save_uploads

st.set_page_config(page_title="University AI Assistant", layout="centered")

UPLOAD_ROOT = PROJECT_ROOT / "data" / "uploads"     # one folder per session
INDEX_ROOT = PROJECT_ROOT / "storage" / "uploads"   # one index per session
SESSION_MAX_AGE_HOURS = 24
_SESSION_FOLDER_NAME = re.compile(r"^[0-9a-f]{32}$")

# Characters that Markdown treats specially. We escape them so that text from
# PDFs, file names or the LLM is shown literally (no links, images or math).
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_~\[\]()#>!|<$:+\-])")


def esc(text):
    """Escape Markdown characters in untrusted text."""
    return _MARKDOWN_SPECIALS.sub(r"\\\1", str(text))


@st.cache_resource(show_spinner=False)
def remove_old_sessions():
    """Delete upload folders of old sessions. Runs once each time the app starts."""
    cutoff = time.time() - SESSION_MAX_AGE_HOURS * 3600
    for root in (UPLOAD_ROOT, INDEX_ROOT):
        if not root.is_dir():
            continue
        for folder in root.iterdir():
            try:
                if (folder.is_dir() and not folder.is_symlink()
                        and _SESSION_FOLDER_NAME.match(folder.name)
                        and folder.stat().st_mtime < cutoff):
                    shutil.rmtree(folder)
            except OSError:
                pass
    return True


def get_session():
    """Return (pipeline, pdf_folder) for this browser session, created once."""
    if "pipeline" not in st.session_state:
        session_id = uuid.uuid4().hex
        pdf_folder = UPLOAD_ROOT / session_id
        retriever = Retriever(pdf_folder=pdf_folder, storage_folder=INDEX_ROOT / session_id)
        st.session_state["pdf_folder"] = pdf_folder
        st.session_state["pipeline"] = RAGPipeline(retriever)
    return st.session_state["pipeline"], st.session_state["pdf_folder"]


def list_session_files(pdf_folder):
    """Names of the PDF files this session has uploaded."""
    if not pdf_folder.is_dir():
        return []
    return sorted(
        p.name for p in pdf_folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )


def handle_upload(pipeline, pdf_folder, files):
    """Save uploaded PDFs, update this session's index, and report what happened."""
    results = save_uploads(files, folder=pdf_folder)

    for result in results:
        if result.status == "rejected":
            st.sidebar.warning(f"{esc(result.original_name)}: not saved. {esc(result.message)}")
        elif result.status == "unchanged":
            st.sidebar.info(f"{esc(result.saved_name)}: {esc(result.message)}")

    saved = [r for r in results if r.status in ("added", "replaced")]
    if not saved:
        return

    st.session_state.pop("last_response", None)   # the documents changed
    with st.spinner("Reading the new documents and updating the index..."):
        report = pipeline.prepare()
    skipped = {item["filename"]: item for item in report.skipped}

    for result in saved:
        label = esc(result.saved_name)
        if result.original_name != result.saved_name:
            label += f" (uploaded as {esc(result.original_name)})"
        if result.saved_name in skipped:
            st.sidebar.warning(
                f"{label}: saved, but it could not be used. "
                f"{esc(skipped[result.saved_name]['message'])}"
            )
        else:
            action = "added" if result.status == "added" else "updated"
            st.sidebar.success(f"{label}: {action} and indexed.")


def render_sidebar(pipeline, pdf_folder):
    """Draw the upload form. Returns the list of documents that can be searched."""
    st.sidebar.header("Add documents")
    with st.sidebar.form("upload_form", clear_on_submit=True):
        files = st.file_uploader(
            f"PDF files (up to {MAX_UPLOAD_MB} MB each)",
            type="pdf",
            accept_multiple_files=True,
        )
        upload_clicked = st.form_submit_button("Save and process")
    st.sidebar.caption(
        "Only the files you upload in this session are used. Refreshing the page "
        "starts over. Your files stay on this computer. When you ask a question, "
        "the question and the most relevant passages are sent to Google's Gemini service."
    )
    if upload_clicked:
        if files:
            handle_upload(pipeline, pdf_folder, files)
        else:
            st.sidebar.warning("Choose at least one PDF file first.")

    with st.spinner("Loading the documents..."):
        documents = pipeline.indexed_documents()
    if documents:
        names = ", ".join(esc(d["filename"]) for d in documents)
        st.sidebar.caption(f"Using: {names}")
    return documents


def render_sources(sources, heading="Sources"):
    if not sources:
        return
    st.subheader(heading)
    for source in sources:
        with st.expander(esc(f"{source.filename} - page {source.page_number}")):
            st.text(source.text)
            st.caption(
                f"Match score: {source.score:.2f}. This is a relative similarity "
                "score, not a probability or a confidence value."
            )


def render_response(response):
    status = response.status
    if status == "answered":
        st.markdown(esc(response.answer))
        if response.citations_verified:
            st.caption("The cited file and page were checked against the retrieved passages.")
        else:
            for message in response.unverified_citations:
                st.warning(esc(message))
            st.warning("Please read the passages below before relying on this answer.")
        render_sources(response.sources)
    elif status == "no_answer":
        st.info(esc(response.answer))
    elif status == "llm_error":
        st.error(esc(response.answer))
        render_sources(response.retrieved[:3], "Closest passages found")
    elif status == "no_documents":
        st.warning("There is nothing to search yet. Upload a PDF file in the sidebar.")
    elif status == "index_error":
        st.error(esc(response.answer))
    else:  # invalid_question
        st.warning(esc(response.answer))


def main():
    remove_old_sessions()
    st.title("University AI Assistant")
    st.caption(
        "Ask a question about the PDF documents you upload. Answers use only "
        "those documents and name the file and page. AI can make mistakes, so "
        "check the cited passages."
    )

    pipeline, pdf_folder = get_session()
    documents = render_sidebar(pipeline, pdf_folder)

    if not documents:
        if list_session_files(pdf_folder):
            st.warning(
                "The files you uploaded contain no readable text, so there is "
                "nothing to search. Try a different PDF."
            )
        else:
            st.info(
                "Upload a PDF file in the sidebar to get started. Answers will "
                "use only the files you upload."
            )
        return

    with st.form("ask_form"):
        question = st.text_input("Your question", max_chars=1000)
        submitted = st.form_submit_button("Ask")

    if submitted:
        with st.spinner("Searching the documents..."):
            st.session_state["last_response"] = pipeline.ask(question)

    response = st.session_state.get("last_response")
    if response is not None:
        st.divider()
        st.markdown(f"**Question:** {esc(response.question)}")
        render_response(response)


try:
    main()
except Exception:
    traceback.print_exc()   # details go to the terminal only, never the browser
    st.error("Something went wrong. Please try again, or restart the app.")