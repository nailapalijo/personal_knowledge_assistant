import hashlib
import tempfile
import uuid
import streamlit as st
import re
from pathlib import Path

from src.document_tasks import generate_quiz, summarize_document
from src.document_loader import MAX_DOCUMENT_SIZE_BYTES
from src.knowledge_base import KnowledgeBase
from src.rag import QUIZ_MODEL_NAME, RAGAssistant
from src.runtime import get_runtime_mode
from src.voice import transcribe_audio


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto"
)


# ---------------------------------
# CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #15112b);
    color: #f4f4f8;
}
[data-testid="stSidebar"] {
    background-color: #11172a;
}
h1, h2, h3 {
    color: #c4b5fd;
}
.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 9px 18px;
}
.stButton > button:hover {
    background: #6d28d9;
    color: white;
}
[data-testid="stMetric"] {
    background: #1b2140;
    padding: 15px;
    border-radius: 12px;
}

.block-container {
    max-width: 1200px;
}

[data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] {
    min-width: 0;
    overflow-wrap: anywhere;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 6rem !important;
    }

    [data-testid="stElementContainer"]:has(.stButton) {
        width: 100%;
    }

    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.75rem !important;
    }

    [data-testid="stColumn"] {
        flex: 1 1 100% !important;
        min-width: 0 !important;
        width: 100% !important;
    }

    .stButton,
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        min-height: 44px;
        width: 100%;
    }

    [data-testid="stFileUploader"],
    [data-testid="stAudioInput"],
    [data-testid="stChatInput"],
    [data-testid="stRadio"] {
        max-width: 100%;
        min-width: 0;
        width: 100%;
    }

    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stAudioInputActionButton"],
    [data-testid="stChatInputSubmitButton"] {
        min-height: 44px;
        min-width: 44px;
    }

    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stChatInputTextArea"] {
        min-height: 44px;
    }

    [data-testid="stSelectbox"] button {
        min-height: 44px;
        min-width: 44px;
    }

    [data-testid="stRadio"] label,
    [data-testid="stFileUploader"] *,
    [data-testid="stChatMessage"] * {
        overflow-wrap: anywhere;
        white-space: normal;
    }

    [data-testid="stRadio"] label {
        min-height: 44px;
    }

    [data-testid="stBottomBlockContainer"] {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

@media (max-width: 480px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    h1 {
        font-size: 1.65rem !important;
        line-height: 1.2;
    }

    h2 {
        font-size: 1.4rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        line-height: 1.5;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# LOAD COMPONENTS
# ---------------------------------

def load_knowledge_base(storage_dir, mode):
    return KnowledgeBase(storage_dir=storage_dir, mode=mode)


@st.cache_resource
def load_ai_assistant(mode):
    return RAGAssistant(mode=mode)


@st.cache_resource
def load_quiz_assistant(mode):
    return RAGAssistant(model_name=QUIZ_MODEL_NAME, mode=mode)


@st.cache_resource
def load_whisper_model(mode):
    from faster_whisper import WhisperModel
    return WhisperModel(
        "tiny" if mode == "cloud" else "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1
    )


RUNTIME_MODE = get_runtime_mode()


# ---------------------------------
# SESSION STATE
# ---------------------------------

defaults = {
    "knowledge_base": None,
    "processed_files": [],
    "chat_history": [],
    "document_summary": "",
    "summary_source": "",
    "quiz_questions": [],
    "quiz_answers": {},
    "quiz_submitted": False,
    "quiz_source": "",
    "quiz_version": 0,
    "last_audio_hash": "",
    "voice_question": "",
    # Increment this value to create a fresh audio_input widget after deletion.
    "voice_widget_version": 0,
    "cloud_storage_id": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------
# VOICE DELETE FUNCTION
# ---------------------------------

def delete_voice_recording():
    # Do not assign to the current audio_input widget key.
    # Incrementing the version gives Streamlit a fresh, empty widget.
    st.session_state.voice_question = ""
    st.session_state.last_audio_hash = ""
    st.session_state.voice_widget_version += 1


# ---------------------------------
# INITIALIZE KNOWLEDGE BASE
# ---------------------------------

if st.session_state.knowledge_base is None:
    try:
        storage_dir = None
        if RUNTIME_MODE == "cloud":
            if not st.session_state.cloud_storage_id:
                st.session_state.cloud_storage_id = uuid.uuid4().hex
            storage_dir = (
                Path(tempfile.gettempdir())
                / "personal_knowledge_assistant"
                / st.session_state.cloud_storage_id
            )
        st.session_state.knowledge_base = load_knowledge_base(
            storage_dir=storage_dir,
            mode=RUNTIME_MODE
        )

        existing_docs = getattr(
            st.session_state.knowledge_base,
            "documents_loaded",
            []
        )

        if isinstance(existing_docs, list):
            st.session_state.processed_files = existing_docs

    except Exception as error:
        st.error(f"Knowledge Base Error: {error}")


# ---------------------------------
# HEADER
# ---------------------------------

st.title("🧠 Personal Knowledge Assistant")

st.markdown(
    "Upload your documents, ask questions, generate summaries "
    "and create quizzes using your personal knowledge base."
)

if RUNTIME_MODE == "cloud":
    st.info(
        "Cloud mode uses CPU-only, document-grounded retrieval. Generative "
        "Qwen models are disabled to stay within free-instance memory limits; "
        "answers quote relevant source text instead. Uploaded documents and "
        "indexes use temporary session storage and may be lost when the app "
        "restarts or the session ends."
    )

st.divider()


# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.header("📁 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        max_upload_size=20,
        help="PDF and TXT files up to 20 MB each. Processing is temporary in Cloud mode."
    )

    if st.button("Process Documents", width="stretch"):

        if not uploaded_files:
            st.warning("Please upload at least one document.")

        elif st.session_state.knowledge_base is None:
            st.error("Knowledge Base initialize nahi hua.")

        else:
            progress = st.progress(0)

            for index, uploaded_file in enumerate(uploaded_files):

                file_path = None

                try:
                    suffix = Path(uploaded_file.name).suffix
                    if suffix.lower() not in {".pdf", ".txt"}:
                        raise ValueError("Only PDF and TXT files are supported.")
                    if uploaded_file.size > MAX_DOCUMENT_SIZE_BYTES:
                        raise ValueError(
                            "Document exceeds the 20 MB processing limit."
                        )

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        file_path = temp_file.name

                    added_chunks = st.session_state.knowledge_base.add_document(
                        str(file_path),
                        source_name=uploaded_file.name
                    )

                    if (
                        uploaded_file.name
                        in st.session_state.knowledge_base.documents_loaded
                        and uploaded_file.name not in st.session_state.processed_files
                    ):
                        st.session_state.processed_files.append(
                            uploaded_file.name
                        )

                    if added_chunks:
                        st.success(f"Processed: {uploaded_file.name}")
                    elif uploaded_file.name in st.session_state.processed_files:
                        st.info(f"Already processed: {uploaded_file.name}")
                    else:
                        st.warning(f"No readable text found: {uploaded_file.name}")

                except Exception as error:
                    st.error(
                        f"{uploaded_file.name}: {error}"
                    )

                finally:
                    if file_path and Path(file_path).exists():
                        Path(file_path).unlink(missing_ok=True)

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.success("Document processing completed!")

    st.divider()

    st.subheader("📚 Processed Documents")

    if st.session_state.processed_files:
        for document in st.session_state.processed_files:
            st.write(f"📄 {document}")
    else:
        st.info("No documents processed yet.")

    st.divider()

    if st.button("🗑️ Clear Chat", width="stretch"):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------
# LOAD ASSISTANT
# ---------------------------------

try:
    assistant = load_ai_assistant(RUNTIME_MODE)
except Exception as error:
    assistant = None
    st.error(f"AI Assistant Error: {error}")


# ---------------------------------
# DASHBOARD
# ---------------------------------

st.subheader("📊 Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Uploaded Documents",
        len(st.session_state.processed_files)
    )

with col2:
    knowledge_base = st.session_state.knowledge_base
    total_chunks = (
        len(knowledge_base.vector_store.text_chunks)
        if knowledge_base is not None
        else 0
    )

    st.metric("Total Chunks", total_chunks)

with col3:
    st.metric(
        "AI Model",
        "Extractive Cloud mode" if RUNTIME_MODE == "cloud" else "Qwen 2.5 · 1.5B"
    )

st.divider()


# ---------------------------------
# DOCUMENT SUMMARIZER
# ---------------------------------

st.header("📝 Document Summarizer")
st.caption(
    "Summaries are extractive: they select sentences from the uploaded "
    "document and are not LLM-generated."
)

if st.session_state.processed_files:

    selected_summary_doc = st.selectbox(
        "Select document",
        st.session_state.processed_files,
        key="summary_document_select"
    )

    if st.button("Generate Summary"):
        try:
            with st.spinner("Generating summary..."):
                summary = summarize_document(
                    selected_summary_doc,
                    st.session_state.knowledge_base
                )

                st.session_state.document_summary = summary
                st.session_state.summary_source = selected_summary_doc
        except Exception as error:
            st.error(f"Summary error: {error}")

    if (
        st.session_state.document_summary
        and st.session_state.summary_source == selected_summary_doc
    ):

        st.subheader(
            f"Summary: {st.session_state.summary_source}"
        )

        st.write(st.session_state.document_summary)

else:
    st.info("Please upload and process a document first.")


st.divider()


# ---------------------------------
# QUIZ GENERATOR
# ---------------------------------

st.header("🎯 Quiz Generator")

if st.session_state.processed_files:

    selected_quiz_doc = st.selectbox(
        "Select document for quiz",
        st.session_state.processed_files,
        key="quiz_document_select"
    )

    if RUNTIME_MODE == "cloud":
        st.caption(
            "Cloud quizzes are built from exact sentences in the selected "
            "document; no generative model is downloaded or run."
        )

    if st.button("Generate Quiz"):
        try:
            status = (
                "Building a source-based quiz..."
                if RUNTIME_MODE == "cloud"
                else "Loading the local quiz model and generating questions..."
            )
            with st.spinner(status):
                quiz_assistant = load_quiz_assistant(RUNTIME_MODE)
                questions = generate_quiz(
                    selected_quiz_doc,
                    st.session_state.knowledge_base,
                    quiz_assistant
                )

            st.session_state.quiz_questions = questions
            st.session_state.quiz_source = selected_quiz_doc
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_version += 1
            if not questions:
                st.warning(
                    "There is not enough distinct, readable source text to "
                    "create a verifiable quiz."
                )
        except Exception as error:
            st.error(f"Quiz generation error: {error}")

    if st.session_state.quiz_questions:

        st.subheader(
            f"Quiz: {st.session_state.quiz_source}"
        )

        for index, question in enumerate(
            st.session_state.quiz_questions
        ):

            st.markdown(
                f"**Q{index + 1}. {question.get('question', 'Question unavailable')}**"
            )

            options = question.get("options", [])

            if options:

                selected_answer = st.radio(
                    "Choose your answer",
                    options,
                    key=f"quiz_{st.session_state.quiz_version}_{index}",
                    index=None
                )

                st.session_state.quiz_answers[index] = selected_answer
                st.caption(f"Difficulty: {question['difficulty']}")

        if st.button("Submit Quiz"):

            st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:

            score = 0

            for index, question in enumerate(
                st.session_state.quiz_questions
            ):

                correct_answer = str(
                    question.get("answer", "")
                ).strip().upper()

                selected = st.session_state.quiz_answers.get(
                    index,
                    ""
                )

                selected_letter = ""

                if selected:
                    match = re.match(
                        r"^\s*([A-D])",
                        selected.upper()
                    )

                    if match:
                        selected_letter = match.group(1)

                if selected_letter == correct_answer:
                    score += 1
                    st.success("Correct answer!")
                else:
                    st.error(
                        f"Incorrect. Correct answer: {correct_answer}"
                    )

                explanation = question.get("explanation")

                if explanation:
                    st.caption(explanation)

            st.subheader(
                f"Your Score: {score}/{len(st.session_state.quiz_questions)}"
            )

else:
    st.info("Upload a document to generate a quiz.")


st.divider()


# ---------------------------------
# RAG CHAT + VOICE INPUT
# ---------------------------------

st.header("💬 Chat with Your Documents")

st.write(
    "Ask questions about your uploaded documents using text or voice."
)

voice_language = st.selectbox(
    "Voice language",
    ["Auto Detect", "English", "Urdu"],
    key="voice_language"
)

language_code = {
    "Auto Detect": None,
    "English": "en",
    "Urdu": "ur"
}[voice_language]

# A new widget key is used after deletion so the old recording disappears.
audio_widget_key = (
    f"voice_audio_input_{st.session_state.voice_widget_version}"
)

audio_file = st.audio_input(
    "🎤 Record your question",
    key=audio_widget_key,
    help=(
        "Requires this browser to support audio recording and have microphone "
        "permission. If recording is unavailable, use the text chat below."
    )
)

st.caption(
    "Voice recognition uses a CPU Whisper model downloaded on first use; "
    "Cloud transcription may be slower. Text chat is available if your "
    "browser blocks microphone access."
)

if audio_file is not None:

    audio_bytes = audio_file.getvalue()

    audio_hash = hashlib.sha256(
        audio_bytes
    ).hexdigest()

    if audio_hash != st.session_state.last_audio_hash:

        try:
            with st.spinner(
                "Loading the CPU voice model if needed and transcribing..."
            ):

                transcript, detected_language = transcribe_audio(
                    audio_bytes,
                    language_code,
                    load_whisper_model(RUNTIME_MODE)
                )

            st.session_state.voice_question = transcript
            st.session_state.last_audio_hash = audio_hash

            if transcript:
                st.success(
                    f"Detected language: {detected_language}"
                )
            else:
                st.warning(
                    "Voice se text recognize nahi hua. Dobara record karein."
                )

        except Exception as error:
            st.session_state.last_audio_hash = audio_hash
            st.error(
                f"Voice recognition error: {error}"
            )

if st.session_state.voice_question:

    st.text_area(
        "Recognized voice text",
        value=st.session_state.voice_question,
        height=90,
        disabled=True,
        key="recognized_voice_display"
    )

if audio_file is not None or st.session_state.voice_question:

    if st.button(
        "🗑️ Delete Recording",
        key="delete_voice_button"
    ):
        delete_voice_recording()
        st.rerun()

typed_question = st.chat_input(
    "Ask a question about your documents..."
)

voice_submit = st.button(
    "🎤 Ask with Voice",
    disabled=not bool(st.session_state.voice_question)
)

question_to_ask = typed_question

if voice_submit:
    question_to_ask = st.session_state.voice_question


# ---------------------------------
# PROCESS USER QUESTION
# ---------------------------------

if question_to_ask:

    if not st.session_state.processed_files:

        st.warning(
            "Please upload and process documents first."
        )

    elif assistant is None:

        st.error(
            "AI Assistant load nahi hua."
        )

    else:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question_to_ask
        })

        try:
            with st.spinner(
                "Searching documents and generating answer..."
            ):

                context = st.session_state.knowledge_base.search_documents(
                    question_to_ask,
                    top_k=5
                )

                answer = assistant.generate_answer(
                    question_to_ask,
                    context
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error generating answer: {error}"
            })

        if voice_submit:
            delete_voice_recording()
        st.rerun()


# ---------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------
# ABOUT PROJECT
# ---------------------------------

st.divider()

with st.expander("ℹ️ About Personal Knowledge Assistant"):

    st.markdown("""
    ### Technologies Used

    - **Python:** Main programming language
    - **Streamlit:** Web application interface
    - **Hugging Face Transformers:** Local language model
    - **Qwen 2.5 1.5B / 0.5B:** Optional, lazy-loaded local answers and quizzes
    - **Multilingual E5-small:** Optional local document and query embeddings
    - **FAISS:** Vector similarity search
    - **RAG:** Retrieval Augmented Generation
    - **Faster Whisper:** On-demand CPU English and Urdu transcription

    ### Features

    - Upload PDF and TXT documents
    - Ask questions from uploaded documents
    - Retrieve relevant document information
    - Generate document summaries
    - Generate multiple-choice quizzes
    - Ask questions through English and Urdu voice input
    - Delete recorded voice and recognized transcript

    """)


# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown(
    """
    <div style="text-align:center; color:#aaa; padding:20px;">
        Built with Python, Streamlit, Hugging Face and FAISS.
    </div>
    """,
    unsafe_allow_html=True
)
