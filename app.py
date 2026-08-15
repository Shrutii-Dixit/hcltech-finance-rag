from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import streamlit as st

from ingest import DATA_DIR, collection_exists, ingest_documents
from rag import REFUSAL_TEXT, answer_question


APP_TITLE = "Finance RAG – HCLTech Financial Report Assistant"
APP_SUBTITLE = "Ask grounded questions about indexed HCLTech quarterly financial reports and review source-backed answers."

REPORTS = [
    {"name": "HCLTech Q1 FY26", "pages": 29},
    {"name": "HCLTech Q2 FY26", "pages": 31},
    {"name": "HCLTech Q3 FY26", "pages": 31},
    {"name": "HCLTech Q4 FY26", "pages": 36},
]


def inject_styles() -> None:
    st.markdown(
        dedent(
            """
            <style>
                .stApp {
                    background: #0f172a;
                    color: #e2e8f0;
                }
                h1, h2, h3, h4 {
                    color: #e2e8f0;
                }
                .subtitle {
                    color: #94a3b8;
                    margin-bottom: 0.75rem;
                }
                .card-row {
                    display: flex;
                    gap: 0.75rem;
                    margin: 0.75rem 0 1rem 0;
                    flex-wrap: wrap;
                }
                .stat-card {
                    background: #111827;
                    border: 1px solid #1f2937;
                    border-radius: 12px;
                    padding: 0.75rem 1rem;
                    min-width: 160px;
                }
                .stat-card .label {
                    font-size: 0.8rem;
                    color: #94a3b8;
                }
                .stat-card .value {
                    font-size: 1.25rem;
                    font-weight: 700;
                    color: #22d3ee;
                }
                .answer-card {
                    background: #111827;
                    border: 1px solid #164e63;
                    border-radius: 12px;
                    padding: 1rem;
                    margin-top: 0.5rem;
                }
                .answer-card-title {
                    color: #67e8f9;
                    font-size: 0.9rem;
                    font-weight: 700;
                    margin-bottom: 0.35rem;
                    letter-spacing: 0.02em;
                }
                .source-meta {
                    color: #bfdbfe;
                    font-size: 0.9rem;
                }
                .report-card {
                    background: #111827;
                    border: 1px solid #1f2937;
                    border-radius: 12px;
                    padding: 0.9rem 1rem;
                    margin-bottom: 0.6rem;
                }
                .report-name {
                    font-weight: 700;
                    color: #e2e8f0;
                }
                .report-pages {
                    color: #93c5fd;
                    font-size: 0.9rem;
                }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )


def ensure_data_directory() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def save_uploaded_pdfs(uploaded_files) -> list[str]:
    saved_files: list[str] = []
    data_dir = ensure_data_directory()
    for uploaded_file in uploaded_files:
        target_path = data_dir / uploaded_file.name
        target_path.write_bytes(uploaded_file.getbuffer())
        saved_files.append(uploaded_file.name)
    return saved_files


def initialize_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "indexed" not in st.session_state:
        st.session_state.indexed = collection_exists()
    if "last_index_stats" not in st.session_state:
        st.session_state.last_index_stats = None
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""


def render_history() -> None:
    if not st.session_state.history:
        st.info("No questions answered yet. Upload PDFs, index them, then ask a question.")
        return

    st.subheader("Previous Questions")
    for index, entry in enumerate(reversed(st.session_state.history), start=1):
        with st.container(border=True):
            st.markdown(f"**Q{index}. {entry['question']}**")
            st.write(entry["answer"])
            sources = entry.get("sources", [])
            if sources:
                st.markdown("**Sources**")
                for source in sources:
                    st.markdown(
                        f"- {source['source']} — Page {source['page']} — {source['quarter']}"
                    )
            else:
                st.write("No supporting sources were returned.")


def render_header() -> None:
    st.title(APP_TITLE)
    st.markdown(f"<p class='subtitle'>{APP_SUBTITLE}</p>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card-row">
            <div class="stat-card"><div class="label">Reports</div><div class="value">4</div></div>
            <div class="stat-card"><div class="label">Pages</div><div class="value">127</div></div>
            <div class="stat-card"><div class="label">Indexed Chunks</div><div class="value">223</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Finance RAG")
        st.caption("HCLTech Financial Report Assistant")
        st.markdown("**Tech stack**")
        st.write("Python | Streamlit | Gemini | ChromaDB | PyPDF")
        st.markdown("**Indexed reports**")
        st.write("4")
        st.info("Answers are grounded in the indexed reports.")
        st.markdown("---")
        st.markdown("**Runtime status**")
        st.write(f"Data folder: {DATA_DIR}")
        st.write(f"Chroma collection ready: {'Yes' if st.session_state.indexed else 'No'}")
        if st.session_state.last_index_stats:
            st.write(f"Files processed: {st.session_state.last_index_stats['files_processed']}")
            st.write(f"Chunks created: {st.session_state.last_index_stats['chunks_created']}")


def render_example_questions() -> None:
    st.markdown("**Examples**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("What was HCLTech's revenue in Q4 FY26?", use_container_width=True):
            st.session_state.question_input = "What was HCLTech's revenue in Q4 FY26?"
            st.rerun()
    with c2:
        if st.button("What was HCLTech's revenue in Q3 FY26?", use_container_width=True):
            st.session_state.question_input = "What was HCLTech's revenue in Q3 FY26?"
            st.rerun()
    with c3:
        if st.button("Compare Q3 FY26 and Q4 FY26 revenue.", use_container_width=True):
            st.session_state.question_input = "Compare Q3 FY26 and Q4 FY26 revenue."
            st.rerun()


def render_reports_tab() -> None:
    st.subheader("Indexed Reports")
    for report in REPORTS:
        st.markdown(
            (
                "<div class='report-card'>"
                f"<div class='report-name'>{report['name']}</div>"
                f"<div class='report-pages'>{report['pages']} pages</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_about_tab() -> None:
    st.subheader("How It Works")
    st.markdown(
        """
PDF Reports
↓
Text Extraction & Chunking
↓
Gemini Embeddings
↓
ChromaDB
↓
Semantic Retrieval
↓
Gemini 2.5 Flash
↓
Grounded Answer + Sources
        """
    )
    st.info("This system is designed to answer only from the indexed HCLTech reports.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_styles()
    initialize_state()
    render_sidebar()
    render_header()

    ask_tab, reports_tab, about_tab = st.tabs(["💬 Ask Questions", "📚 Reports", "ℹ️ About"])

    with ask_tab:
        st.subheader("Document Setup")
        uploaded_files = st.file_uploader(
            "Upload one or more HCLTech quarterly PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s). They will be saved into the data folder before indexing.")

        index_button = st.button("Index Documents", type="primary")
        if index_button:
            try:
                if uploaded_files:
                    saved = save_uploaded_pdfs(uploaded_files)
                    st.success(f"Saved files to data/: {', '.join(saved)}")
                with st.spinner("Indexing documents and creating embeddings..."):
                    stats = ingest_documents()
                st.session_state.indexed = True
                st.session_state.last_index_stats = stats
                st.success(
                    f"Indexed {stats['files_processed']} files and created {stats['chunks_created']} chunks."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        st.subheader("Ask a Question")
        render_example_questions()
        question = st.text_area(
            "Enter your question",
            key="question_input",
            height=120,
            placeholder="Example: What was the revenue in the latest quarter?",
        )
        ask_button = st.button("Ask", disabled=not st.session_state.indexed)

        if not st.session_state.indexed:
            st.warning("Please index the documents first. If the ChromaDB collection is missing, upload the PDFs and click Index Documents.")

        if ask_button:
            if not st.session_state.indexed:
                st.warning("The database is not indexed yet. Upload PDFs and click Index Documents first.")
            elif not question.strip():
                st.warning("Please enter a question before asking.")
            else:
                try:
                    with st.spinner("Generating a grounded answer..."):
                        result = answer_question(question.strip())
                    st.session_state.history.append(
                        {
                            "question": question.strip(),
                            "answer": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                    if result["answer"] == REFUSAL_TEXT:
                        st.warning(result["answer"])
                    st.markdown(
                        (
                            "<div class='answer-card'>"
                            "<div class='answer-card-title'>ANSWER</div>"
                            f"<div>{result['answer']}</div>"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

                    st.markdown("### Sources")
                    if result["sources"]:
                        for source in result["sources"]:
                            title = f"{source['source']} — Page {source['page']}"
                            with st.expander(title):
                                st.markdown(
                                    f"<div class='source-meta'>Quarter: {source['quarter']}</div>",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.write("No sources were returned for this answer.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        st.divider()
        render_history()

    with reports_tab:
        render_reports_tab()

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()