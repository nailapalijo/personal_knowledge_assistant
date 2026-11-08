
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

/* Main background */

.stApp {
    background: linear-gradient(135deg, #0b1020, #17132e);
    color: #F8FAFC;
}

/* Main container */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */

h1, h2, h3 {
    color: #A78BFA !important;
}

/* General text visibility */

.stApp p,
.stApp li,
.stApp label,
.stApp [data-testid="stCaptionContainer"] {
    color: #F8FAFC;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #6d4aff, #8b5cf6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 600;
    width: 100%;
    transition: background 0.2s ease;
    touch-action: manipulation;
    cursor: pointer;
}

.stButton > button p,
.stFormSubmitButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #9b7bff !important;
    border: none !important;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #111827;
}

/* Text input */

.stTextInput input {
    color: #F8FAFC !important;
    background-color: #172033 !important;
    border-radius: 10px;
    min-height: 48px;
    border: 1px solid #64748b !important;
    font-size: 16px !important;
}

.stTextInput input::placeholder {
    color: #CBD5E1 !important;
    opacity: 1;
}

/* File uploader */

[data-testid="stFileUploader"] {
    border: 1px dashed #8b5cf6;
    border-radius: 12px;
    padding: 10px;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background: #172033 !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #F8FAFC !important;
}

/* Markdown answer text */

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #F8FAFC !important;
}

/* Metrics */

[data-testid="stMetric"] {
    background: rgba(139,92,246,0.10);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Expanders */

[data-testid="stExpander"] {
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #F8FAFC !important;
}

/* Mobile responsive */

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

    .stButton > button {
        width: 100% !important;
        min-height: 50px !important;
        font-size: 16px !important;
    }

    .stTextInput input {
        min-height: 50px !important;
        font-size: 16px !important;
    }

    [data-testid="stChatMessage"] {
        padding: 12px;
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
# RESTORE KNOWLEDGE BASE
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
    "and ask questions using your AI model."
)

st.divider()


# ---------------------------------
# SIDEBAR
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
        use_container_width=True,
        key="process_documents_button"
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

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
        key="clear_chat_button"
    ):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------
# PROCESS DOCUMENTS
# ---------------------------------

if process_button:

    if not uploaded_files:

        st.warning("Please upload a PDF or TXT file first.")

    else:

        with st.spinner("Processing documents..."):

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

                    file_path.write_bytes(uploaded_file.getvalue())

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

st.write("Ask anything related to your uploaded documents.")


# ---------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------
# QUESTION INPUT
# ---------------------------------

question = st.text_input(
    "Your question",
    placeholder="Example: What is my educational background?",
    key="question_input"
)

ask_button = st.button(
    "Ask AI 🚀",
    use_container_width=True,
    key="ask_ai_button",
    type="primary"
)


# ---------------------------------
# IMPROVE SEARCH QUERY
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

                search_query = improve_search_query(question)

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

                    if not answer or not str(answer).strip():
                        answer = "The AI model returned an empty response. Please try again."

                else:

                    answer = (
                        "I could not find relevant information "
                        "in your uploaded documents."
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": str(answer)
                })

            except Exception as error:

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"AI Error: {error}"
                })

        st.rerun()


# ---------------------------------
# ABOUT PROJECT
# ---------------------------------

st.divider()

with st.expander("ℹ️ About This Project"):

    st.write("""
    Personal Knowledge Assistant is an AI-powered application.

    **Features:**

    - PDF and TXT document upload
    - Automatic text extraction
    - Text chunking
    - Hugging Face embeddings
    - FAISS vector database
    - Semantic document search
    - Retrieval-Augmented Generation (RAG)
    - Local FLAN-T5 language model

    **Privacy:** The application uses its configured AI model.
    No paid OpenAI API is required.
    """)

st.caption(
    "Built with Python, Streamlit, Hugging Face and FAISS."
)
