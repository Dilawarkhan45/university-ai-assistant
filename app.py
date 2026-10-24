"""University AI Assistant - Streamlit interface (questions and PDF upload).

Run from the project root:   python -m streamlit run app.py
"""

import re
import traceback

import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.upload_manager import MAX_UPLOAD_MB, save_uploads

st.set_page_config(page_title="University AI Assistant", layout="centered")

# Characters that Markdown treats specially. We escape them so that text from
# PDFs, file names or the LLM is shown literally (no links, images or math).
_MARKDOWN_SPECIALS = re.compile(r"([\\`*_~\[\]()#>!|<$:+\-])")


def esc(text):
    """Escape Markdown characters in untrusted text."""
    return _MARKDOWN_SPECIALS.sub(r"\\\1", str(text))


@st.cache_resource(show_spinner=False)
def get_pipeline():
    """Create the pipeline once and reuse it across reruns."""
    return RAGPipeline()


def handle_upload(pipeline, files):
    """Save uploaded PDFs, update the index, and report what happened."""
    results = save_uploads(files)

    for result in results:
        if result.status == "rejected":
            st.sidebar.warning(f"{esc(result.original_name)}: not saved. {esc(result.message)}")
        elif result.status == "unchanged":
            st.sidebar.info(f"{esc(result.saved_name)}: {esc(result.message)}")

    saved = [r for r in results if r.status in ("added", "replaced")]
    if not saved:
        return

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


def render_sidebar(pipeline):
    st.sidebar.header("Documents")

    if st.sidebar.button("Check for document changes"):
        with st.spinner("Checking the documents..."):
            report = pipeline.prepare()
        if report.rebuilt:
            st.sidebar.success("Documents changed, so the index was rebuilt.")
        else:
            st.sidebar.info("The index is already up to date.")

    if st.sidebar.button("Rebuild index"):
        with st.spinner("Rebuilding the index. This can take a minute..."):
            pipeline.rebuild()
        st.sidebar.success("The index was rebuilt.")

    st.sidebar.subheader("Add documents")
    with st.sidebar.form("upload_form", clear_on_submit=True):
        files = st.file_uploader(
            f"PDF files (up to {MAX_UPLOAD_MB} MB each)",
            type="pdf",
            accept_multiple_files=True,
        )
        upload_clicked = st.form_submit_button("Save and process")
    st.sidebar.caption(
        "Uploaded files stay on this computer. When you ask a question, the "
        "question and the most relevant passages are sent to Google's Gemini service."
    )
    if upload_clicked:
        if files:
            handle_upload(pipeline, files)
        else:
            st.sidebar.warning("Choose at least one PDF file first.")

    # The list is drawn last, so it always shows the latest state.
    with st.spinner("Loading the documents..."):
        documents = pipeline.indexed_documents()

    if not documents:
        st.sidebar.warning("No documents are indexed yet. Add a PDF file above.")
    else:
        st.sidebar.markdown(f"**{len(documents)} document(s) indexed**")
        for document in documents:
            st.sidebar.markdown(
                f"- {esc(document['filename'])} "
                f"({document['pages_with_text']} page(s), {document['chunks']} passage(s))"
            )

    report = pipeline.retriever.report
    skipped = report.skipped if report else []
    if skipped:
        with st.sidebar.expander(f"{len(skipped)} file(s) could not be used"):
            for item in skipped:
                st.markdown(f"- {esc(item['filename'])}: {esc(item['message'])}")


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
    elif status in ("index_error", "no_documents"):
        st.error(esc(response.answer))
    else:  # invalid_question
        st.warning(esc(response.answer))


def main():
    st.title("University AI Assistant")
    st.caption(
        "Ask a question about the university documents. Answers use only those "
        "documents and name the file and page. AI can make mistakes, so check "
        "the cited passages."
    )

    pipeline = get_pipeline()
    render_sidebar(pipeline)

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