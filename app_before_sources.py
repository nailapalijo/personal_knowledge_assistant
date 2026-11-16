
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
# LOAD AI MODELS
# ---------------------------------

@st.cache_resource
def load_knowledge_base():
    return KnowledgeBase()


@st.cache_resource
def load_ai_assistant():
    return RAGAssistant()


# ---------------------------------
# SESSION STATE
# ---------------------------------

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
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
    "Your private AI assistant. Upload your documents "
    "and ask questions using your local AI model."
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

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------
# PROCESS DOCUMENTS
# ---------------------------------

if process_button:

    if not uploaded_files:

        st.warning("Please upload a PDF or TXT file first.")

    else:

        with st.spinner("Loading AI and processing documents..."):

            try:

                if st.session_state.knowledge_base is None:
                    st.session_state.knowledge_base = load_knowledge_base()

                upload_folder = Path("data/uploads")
                upload_folder.mkdir(parents=True, exist_ok=True)

                for uploaded_file in uploaded_files:

                    if uploaded_file.name in st.session_state.processed_files:
                        st.info(
                            f"{uploaded_file.name} is already processed."
                        )
                        continue

                    file_path = upload_folder / uploaded_file.name

                    file_path.write_bytes(
                        uploaded_file.getvalue()
                    )

                    try:

                        chunks = (
                            st.session_state.knowledge_base.add_document(
                                str(file_path)
                            )
                        )

                        if chunks > 0:

                            st.session_state.processed_files.append(
                                uploaded_file.name
                            )

                            st.success(
                                f"{uploaded_file.name} processed "
                                f"successfully! ({chunks} chunks)"
                            )

                        else:

                            st.warning(
                                f"No readable text found in "
                                f"{uploaded_file.name}."
                            )

                    except Exception as error:

                        st.error(
                            f"Error processing {uploaded_file.name}: {error}"
                        )

            except Exception as error:

                st.error(f"Knowledge Base Error: {error}")


# ---------------------------------
# MAIN CHAT AREA
# ---------------------------------

st.divider()

st.header("💬 Ask Your Documents")

st.write(
    "Ask anything related to your uploaded documents."
)


# ---------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.write(message["content"])


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
# QUESTION-BASED SEARCH
# ---------------------------------

def improve_search_query(question):

    q = question.lower()

    if any(word in q for word in ["name", "who am i"]):

        return "Naila Palijo full name personal profile"

    elif any(word in q for word in [
        "university", "graduate", "graduated"
    ]):

        return (
            "Naila Palijo graduated from Mehran University "
            "of Engineering and Technology MUET Jamshoro"
        )

    elif any(word in q for word in [
        "degree", "education", "educational"
    ]):

        return (
            "Naila Palijo Bachelor Computer Systems Engineering "
            "Mehran University Jamshoro education"
        )

    elif any(word in q for word in [
        "skill", "technical"
    ]):

        return (
            "Naila Palijo technical skills Python Django Java "
            "JavaScript React HTML CSS Git"
        )

    else:

        return question


# ---------------------------------
# GENERATE ANSWER
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

                # Improve query according to question type
                search_query = improve_search_query(question)

                # Retrieve relevant document chunks
                results = (
                    st.session_state.knowledge_base.search_documents(
                        search_query,
                        top_k=3
                    )
                )

                if results:

                    context = "\n\n".join(
                        item["text"] for item in results
                    )

                    assistant = load_ai_assistant()

                    answer = assistant.generate_answer(
                        question,
                        context
                    )

                else:

                    answer = (
                        "I could not find relevant information "
                        "in your uploaded documents."
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:

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
    - Hugging Face embeddings
    - FAISS vector database
    - Semantic document search
    - Retrieval-Augmented Generation (RAG)
    - Local FLAN-T5 language model

    **Privacy:** AI models run locally on your computer.
    No paid OpenAI API is required.
    """)

st.caption(
    "Built with Python, Streamlit, Hugging Face and FAISS."
)
