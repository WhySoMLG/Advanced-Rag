import streamlit as st
import tempfile
import os
from datetime import datetime, time, timezone
from pathlib import Path

# Import your top-level façade
from rag_connector import RAGConnector

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Multimodal RAG", page_icon="🧠", layout="wide")
st.title("🧠 Multimodal RAG Assistant")
st.markdown("Upload documents, images, audio, or video, and ask questions about them!")

# --- 2. SESSION STATE SETUP ---
# Initialize the RAG connector once so it doesn't reload on every UI interaction
if "rag" not in st.session_state:
    with st.spinner("Initializing Local Models and Vector DB..."):
        st.session_state.rag = RAGConnector()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

rag: RAGConnector = st.session_state.rag

# --- 3. SIDEBAR: DATA INGESTION + FILTERS ---
with st.sidebar:
    st.header("📥 Ingest Data")
    st.info("Supported: PDFs, DOCX, Images, Video, Audio, and Code.")

    uploaded_file = st.file_uploader("Upload a file to the knowledge base:")

    if uploaded_file is not None:
        if st.button("Process & Index File"):
            with st.spinner(f"Ingesting `{uploaded_file.name}` (This might take a while for video/audio)..."):
                # Save the uploaded file temporarily so the pipeline can read it from disk
                file_extension = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # Run your ingestion and indexing pipeline
                    rag.index(tmp_path)
                    st.success(f"✅ Successfully indexed: {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                finally:
                    # Clean up the temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    st.divider()

    # ── Retrieval Filters ────────────────────────────────────────────────
    st.header("🔍 Retrieval Filters")
    st.caption("Restrict which chunks the retriever can return.")

    # Populate dropdowns from the live index. These are cheap reads (one
    # ChromaDB .get() each) — no need to cache aggressively.
    available_modalities = rag.list_modalities()
    available_sources    = rag.list_sources()

    selected_modalities = st.multiselect(
        "Modality",
        options=available_modalities,
        default=[],
        help="Empty = all modalities."
    )
    selected_sources = st.multiselect(
        "Source file",
        options=available_sources,
        default=[],
        help="Empty = all files."
    )

    use_date_filter = st.checkbox(
        "Filter by indexed date",
        value=False,
        help="Only retrieve chunks indexed in this date range.",
    )
    date_from_ts = None
    date_to_ts   = None
    if use_date_filter:
        col_from, col_to = st.columns(2)
        with col_from:
            d_from = st.date_input("From", value=None, key="date_from")
        with col_to:
            d_to = st.date_input("To", value=None, key="date_to")
        if d_from:
            date_from_ts = int(datetime.combine(d_from, time.min, tzinfo=timezone.utc).timestamp())
        if d_to:
            date_to_ts = int(datetime.combine(d_to, time.max, tzinfo=timezone.utc).timestamp())

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 4. MAIN CHAT INTERFACE ---
# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new questions
if prompt := st.chat_input("Ask a question based on your uploaded data..."):

    # Show user question instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the RAG response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and reasoning..."):
            try:
                # Query the backend with the active filters and chat history.
                # messages[:-1] = everything BEFORE the current question, so the
                # rewriter can resolve pronouns in follow-ups.
                results = rag.query(
                    question=prompt,
                    top_k=5,
                    modality_filter=selected_modalities or None,
                    source_filter=selected_sources or None,
                    date_from_ts=date_from_ts,
                    date_to_ts=date_to_ts,
                    chat_history=st.session_state.messages[:-1],
                )
                answer  = results["answer"]
                sources = results["sources"]

                # Render the answer
                st.markdown(answer)

                # Render the sources in an expander
                if sources:
                    # Highlight when multiple files contributed to the answer.
                    unique_files = {
                        s.get("metadata", {}).get("source", "?") for s in sources
                    }
                    header = (
                        f"📚 Sources used ({len(sources)} chunks across "
                        f"{len(unique_files)} file{'s' if len(unique_files) != 1 else ''})"
                    )
                    with st.expander(header):
                        # If the question was rewritten for retrieval, show it.
                        search_query = results.get("search_query", prompt)
                        if search_query != prompt:
                            st.caption(f"🔄 **Rewritten query:** _{search_query}_")
                        for i, src in enumerate(sources, 1):
                            meta        = src.get("metadata", {})
                            source_name = meta.get("source", "Unknown File")
                            modality    = meta.get("modality", "Unknown")
                            indexed_at  = meta.get("indexed_at", "")
                            doc_text    = src.get("document", "")
                            rerank      = src.get("rerank_score")

                            # Header row: matches the [N] citation in the answer.
                            score_bits = [f"RRF `{src['rrf_score']:.3f}`"]
                            if rerank is not None:
                                score_bits.append(f"Rerank `{rerank:.3f}`")
                            st.markdown(
                                f"**[{i}] `{source_name}`** — _{modality}_  \n"
                                f"<span style='color:#888;font-size:0.85em'>"
                                f"{' · '.join(score_bits)}"
                                f"{' · indexed ' + indexed_at if indexed_at else ''}"
                                f"</span>",
                                unsafe_allow_html=True,
                            )
                            # Show a short snippet of the chunk so the citation is verifiable.
                            if doc_text:
                                snippet = doc_text.strip().replace("\n", " ")
                                if len(snippet) > 350:
                                    snippet = snippet[:350].rsplit(" ", 1)[0] + "…"
                                st.caption(snippet)
                            st.write("")  # spacing between entries

                # Save to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error during retrieval: {e}")
