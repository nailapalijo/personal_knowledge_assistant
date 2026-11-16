
import streamlit as st
from pathlib import Path

from src.knowledge_base import KnowledgeBase
from src.rag import RAGAssistant


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------
# CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #17132e);
    color: #ffffff;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #a78bfa !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #6d4aff, #8b5cf6);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 600;
    width: 100%;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #9b7bff;
    border: none;
    transform: translateY(-1px);
}

[data-testid="stSidebar"] {
    background: #111827;
}

.stTextInput input {
    border-radius: 10px;
    min-height: 44px;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #8b5cf6;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}

[data-testid="stMetric"] {
    background: rgba(139,92,246,0.10);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 12px;
}

@media screen and (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem 0.8rem;
    }

    h1 {
        font-size: 1.65rem !important;
        line-height: 1.3 !important;
    }

    h2 {
        font-size: 1.35rem !important;
    }

    h3 {
        font-size: 1.1rem !important;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        width: 100%;
        min-height: 46px;
    }

    [data-testid="stChatMessage"] {
        padding: 10px;
    }
}

@media screen and (max-width: 480px) {
    .block-container {
        padding-left: 0.6rem;
        padding-right: 0.6rem;
    }

    h1 {
        font-size: 1.45rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# LOAD LOCAL AI MODELS
# ---------------------------------

@st.cache_resource
def load_knowledge_base():
    return KnowledgeBase()


@st.cache_resource
def load_ai_assistant():
    return RAGAssistant()


# ---------------------------------
# DOCUMENT SUMMARIZER FUNCTION
# ---------------------------------

def summarize_document(chunks, assistant):

    if not chunks:
        return "No readable text found in this document."

    summaries = []

    # First pass: summarize individual chunks
    for index, chunk in enumerate(chunks):

        summary = assistant.generate_answer(
            "Summarize the main information in this text. "
            "Keep important facts, names, dates and details. "
            "Use simple and clear language.",
            chunk
        )

        if summary.strip():
            summaries.append(summary.strip())

    if not summaries:
        return "Unable to generate a summary."

    # Reduce summaries in small groups
    while len(summaries) > 3:

        reduced = []

        for i in range(0, len(summaries), 3):

            group = "\n\n".join(summaries[i:i + 3])

            result = assistant.generate_answer(
                "Combine these notes into a shorter summary. "
                "Preserve important information and avoid repetition.",
                group
            )

            if result.strip():
                reduced.append(result.strip())

        if not reduced:
            break

        summaries = reduced

    # Final summary
    final_context = "\n\n".join(summaries)

    return assistant.generate_answer(
        "Create one clear, well-organized final summary. "
        "Include the main topic, key points and important details. "
        "Do not add information that is not in the text.",
        final_context
    )


# ---------------------------------
# SESSION STATE
# ---------------------------------

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_summary" not in st.session_state:
    st.session_state.document_summary = ""

if "summary_source" not in st.session_state:
    st.session_state.summary_source = ""


# ---------------------------------
# CLEAR CHAT CALLBACK
# ---------------------------------

def clear_chat():
    st.session_state.chat_history = []


# ---------------------------------
# RESTORE SAVED KNOWLEDGE BASE
# ---------------------------------

if st.session_state.knowledge_base is None:

    try:
        st.session_state.knowledge_base = load_knowledge_base()

        st.session_state.processed_files = list(
            st.session_state.knowledge_base.documents_loaded
        )

    except Exception as error:
        st.error(f"Knowledge Base loading error: {error}")


# ---------------------------------
# HEADER
# ---------------------------------

st.title("🧠 Personal Knowledge Assistant")

st.write(
    "Your personal AI assistant. Upload your documents "
    "and ask questions using a locally running Hugging Face model."
)

st.divider()


# ---------------------------------
# SIDEBAR - DOCUMENT LIBRARY
# ---------------------------------

with st.sidebar:

    st.header("📚 Document Library")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "⚡ Process Documents",
        use_container_width=True
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.metric(
        "Processed Documents",
        len(st.session_state.processed_files)
    )

    if st.session_state.processed_files:

        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")

    else:
        st.caption("No documents processed yet.")

    st.divider()

    st.button(
        "🗑️ Clear Chat",
        key="clear_chat_button",
        on_click=clear_chat,
        use_container_width=True
    )


# ---------------------------------
# PROCESS DOCUMENTS
# ---------------------------------

if process_button:

    if not uploaded_files:

        st.warning("Please upload a PDF or TXT file first.")

    elif st.session_state.knowledge_base is None:

        st.error(
            "Knowledge Base is not available. "
            "Check the loading error above."
        )

    else:

        upload_folder = Path("data/uploads")
        upload_folder.mkdir(parents=True, exist_ok=True)

        with st.spinner("Processing your documents..."):

            for uploaded_file in uploaded_files:

                safe_filename = Path(uploaded_file.name).name

                if safe_filename in st.session_state.processed_files:
                    st.info(f"{safe_filename} is already processed.")
                    continue

                file_path = upload_folder / safe_filename

                try:

                    file_path.write_bytes(
                        uploaded_file.getvalue()
                    )

                    chunks = (
                        st.session_state.knowledge_base.add_document(
                            str(file_path)
                        )
                    )

                    if chunks and chunks > 0:

                        if safe_filename not in st.session_state.processed_files:

                            st.session_state.processed_files.append(
                                safe_filename
                            )

                        st.success(
                            f"{safe_filename} processed successfully! "
                            f"({chunks} chunks)"
                        )

                    else:

                        st.warning(
                            f"No readable text found in {safe_filename}."
                        )

                except Exception as error:

                    st.error(
                        f"Error processing {safe_filename}: {error}"
                    )


# ---------------------------------
# DOCUMENT SUMMARIZER
# ---------------------------------

st.divider()

st.header("📝 Document Summarizer")

st.write(
    "Generate a summary of any document in your Knowledge Base."
)

if st.session_state.processed_files:

    selected_document = st.selectbox(
        "Select a document to summarize",
        st.session_state.processed_files,
        key="summary_document"
    )

    if st.button(
        "✨ Generate Summary",
        key="generate_summary_button",
        use_container_width=True
    ):

        try:

            knowledge_base = st.session_state.knowledge_base

            vector_store = knowledge_base.vector_store

            document_chunks = [

                text

                for text, source in zip(
                    vector_store.text_chunks,
                    vector_store.sources
                )

                if Path(source).name == selected_document
            ]

            if not document_chunks:

                st.warning(
                    "No text chunks found for this document."
                )

            else:

                with st.spinner(
                    "AI is reading and summarizing your document..."
                ):

                    assistant = load_ai_assistant()

                    summary = summarize_document(
                        document_chunks,
                        assistant
                    )

                    st.session_state.document_summary = summary

                    st.session_state.summary_source = selected_document

        except Exception as error:

            st.error(f"Summarization error: {error}")


else:

    st.info(
        "Upload and process a document to use the summarizer."
    )


# Display generated summary
if st.session_state.document_summary:

    st.subheader(
        f"Summary: {st.session_state.summary_source}"
    )

    st.markdown(
        st.session_state.document_summary
    )

    st.download_button(
        "⬇️ Download Summary",
        data=st.session_state.document_summary,
        file_name="document_summary.txt",
        mime="text/plain",
        use_container_width=True
    )

st.divider()


# ---------------------------------
# MAIN CHAT AREA
# ---------------------------------

st.header("💬 Ask Your Documents")

st.write(
    "Ask questions about the information in your uploaded documents."
)


# ---------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------
# QUESTION INPUT
# ---------------------------------

with st.form("question_form", clear_on_submit=True):

    question = st.text_input(
        "Your question",
        placeholder="Example: What is my educational background?"
    )

    ask_button = st.form_submit_button(
        "Ask AI 🚀",
        use_container_width=True
    )


# ---------------------------------
# GENERATE ANSWER USING RAG
# ---------------------------------

if ask_button:

    if not question.strip():

        st.warning("Please type a question first.")

    elif (
        st.session_state.knowledge_base is None
        or not st.session_state.processed_files
    ):

        st.warning(
            "Please upload and process a document first."
        )

    else:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner(
            "Searching documents and generating your answer..."
        ):

            try:

                search_query = question.strip()

                results = (
                    st.session_state.knowledge_base.search_documents(
                        search_query,
                        top_k=5
                    )
                )

                if results:

                    context = "\n\n".join(
                        item["text"]
                        for item in results
                        if item.get("text")
                    )

                    if not context.strip():

                        answer = (
                            "I could not find readable information "
                            "in the retrieved document sections."
                        )

                    else:

                        assistant = load_ai_assistant()

                        answer = assistant.generate_answer(
                            question,
                            context
                        )

                        sources = list(dict.fromkeys(
                            item.get("source", "Unknown document")
                            for item in results
                            if item.get("source")
                        ))

                        if sources:

                            answer += "\n\n**Sources:**\n"

                            answer += "\n".join(
                                f"- {source}"
                                for source in sources
                            )

                else:

                    answer = (
                        "I could not find relevant information "
                        "in your uploaded documents. "
                        "Try asking your question differently."
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": (
                        "Sorry, an error occurred while generating "
                        "your answer. Please try again."
                    )
                })

                st.error(f"AI Error: {error}")

        st.rerun()


# ---------------------------------
# ABOUT PROJECT
# ---------------------------------

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write("""
    Personal Knowledge Assistant is a local AI-powered application.

    **Features:**

    - PDF and TXT document upload
    - Automatic text extraction
    - Text chunking
    - Hugging Face sentence embeddings
    - FAISS vector database
    - Semantic document search
    - Retrieval-Augmented Generation (RAG)
    - Local FLAN-T5 language model
    - Document source references
    - Document summarization
    - Summary download
    - Mobile-friendly interface

    **Privacy:**

    AI models run locally on your computer.
    No paid AI API is used.
    """)

st.caption(
    "Built with Python, Streamlit, Hugging Face and FAISS."
)
