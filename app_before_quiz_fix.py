import streamlit as st
import json
import re
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
    background: linear-gradient(135deg, #090b18 0%, #11132b 55%, #17102d 100%);
    color: #f4f4ff;
}
[data-testid="stSidebar"] {
    background: #101225;
    border-right: 1px solid #29264b;
}
h1, h2, h3 {
    color: #ffffff !important;
}
p, label, li {
    color: #d6d7ed;
}
.stButton > button {
    background: linear-gradient(90deg, #6955df, #9b55db);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.1rem;
    font-weight: 600;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #806cf0, #b66af0);
    color: white;
    border: none;
}
.stTextInput input, .stTextArea textarea {
    background: #171a32;
    color: white;
    border: 1px solid #45416e;
    border-radius: 8px;
}
div[data-testid="stFileUploader"] {
    background: #171a32;
    border: 1px dashed #7164bd;
    border-radius: 12px;
    padding: 12px;
}
div[data-testid="stMetric"] {
    background: #171a32;
    border: 1px solid #302d55;
    padding: 15px;
    border-radius: 12px;
}
hr {
    border-color: #343052;
}
.small-note {
    color: #aaa8c8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# CACHED LOADERS
# ---------------------------------

@st.cache_resource
def load_knowledge_base():
    return KnowledgeBase()


@st.cache_resource
def load_ai_assistant():
    return RAGAssistant()


# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------

def get_document_chunks(document_name, knowledge_base):
    """Return all chunks belonging to the selected document."""

    vector_store = knowledge_base.vector_store

    chunks = getattr(vector_store, "text_chunks", [])
    sources = getattr(vector_store, "sources", [])

    document_chunks = []

    for chunk, source in zip(chunks, sources):
        if source == document_name and isinstance(chunk, str):
            if chunk.strip():
                document_chunks.append(chunk.strip())

    return document_chunks


def summarize_document(chunks, assistant):
    """
    Generate a document summary using one model call.
    This is faster than generating a separate summary for every chunk.
    """

    if not chunks:
        return "No readable text found in this document."

    clean_chunks = [
        chunk.strip()
        for chunk in chunks
        if isinstance(chunk, str) and chunk.strip()
    ]

    if not clean_chunks:
        return "No readable text found in this document."

    context = "\n\n".join(clean_chunks)

    # Limit context to avoid very long CPU generation time.
    max_context_chars = 6500
    was_truncated = len(context) > max_context_chars
    context = context[:max_context_chars]

    instruction = """
You are a professional document summarizer.

Create a clear and well-organized summary using ONLY
the supplied document.

Use these sections:

1. Main Topic
Explain what the document is about.

2. Key Points
List the most important information.

3. Important Details
Include names, dates, facts and figures when available.

4. Conclusion
Explain the main takeaway.

Rules:
- Use only the supplied document.
- Never invent facts or information.
- Do not add outside knowledge.
- Use simple and clear English.
- Use headings and bullet points.
- Avoid unnecessary repetition.
- If information is missing, do not guess.
"""

    try:
        summary = assistant.generate_content(
            instruction,
            context,
            max_new_tokens=350
        )

        if not summary or not summary.strip():
            return "Unable to generate a summary. Please try again."

        summary = summary.strip()

        if was_truncated:
            summary += (
                "\n\n---\n"
                "*Note: This document was lengthy. "
                "The summary uses the first 6,500 characters "
                "of its extracted text.*"
            )

        return summary

    except Exception as error:
        return f"Summary generation failed: {error}"


def generate_quiz(context, number_of_questions, assistant):
    """Generate MCQs from the supplied document context."""

    instruction = f"""
Create exactly {number_of_questions} multiple-choice questions
from the supplied document.

Return ONLY valid JSON. Do not use Markdown code fences.

Use this exact JSON structure:

{{
  "questions": [
    {{
      "question": "Question text",
      "options": {{
        "A": "Option one",
        "B": "Option two",
        "C": "Option three",
        "D": "Option four"
      }},
      "answer": "A",
      "explanation": "Short explanation based on the document"
    }}
  ]
}}

Rules:
- Use only information from the document.
- Each question must have four options.
- Only one option must be correct.
- The answer must be A, B, C, or D.
- Do not invent facts.
- Keep questions clear and useful.
"""

    return assistant.generate_content(
        instruction,
        context,
        max_new_tokens=500
    )


def parse_quiz(raw_output):
    """Convert the model's JSON output into quiz questions."""

    if not raw_output:
        return []

    cleaned = raw_output.strip()

    # Remove Markdown code fences if the model adds them.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract a JSON object from extra model text.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)

        if not match:
            return []

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    questions = data.get("questions", [])

    valid_questions = []

    for item in questions:
        question = item.get("question")
        options = item.get("options", {})
        answer = str(item.get("answer", "")).strip().upper()
        explanation = item.get("explanation", "")

        if (
            question
            and isinstance(options, dict)
            and all(key in options for key in ["A", "B", "C", "D"])
            and answer in ["A", "B", "C", "D"]
        ):
            valid_questions.append({
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": explanation
            })

    return valid_questions


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

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_source" not in st.session_state:
    st.session_state.quiz_source = ""


# ---------------------------------
# LOAD KNOWLEDGE BASE
# ---------------------------------

try:
    if st.session_state.knowledge_base is None:
        st.session_state.knowledge_base = load_knowledge_base()

    knowledge_base = st.session_state.knowledge_base

    # Load previously processed document names.
    st.session_state.processed_files = list(
        getattr(knowledge_base, "documents_loaded", [])
    )

except Exception as error:
    st.error(f"Knowledge Base loading failed: {error}")
    st.stop()


# ---------------------------------
# HEADER
# ---------------------------------

st.title("🧠 Personal Knowledge Assistant")

st.markdown(
    "Your personal AI assistant. Upload your documents and ask "
    "questions using a locally running Hugging Face model."
)

st.caption(
    "Upload documents • Generate summaries • Ask questions • Practice with quizzes"
)

st.divider()


# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.title("🧠 Knowledge Base")

    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("📥 Process Documents", use_container_width=True):

        if not uploaded_files:
            st.warning("Please upload at least one PDF or TXT file.")

        else:
            upload_directory = Path("data/uploads")
            upload_directory.mkdir(parents=True, exist_ok=True)

            progress = st.progress(0)
            status = st.empty()

            total_files = len(uploaded_files)

            for index, uploaded_file in enumerate(uploaded_files):

                file_path = upload_directory / Path(uploaded_file.name).name

                try:
                    file_path.write_bytes(uploaded_file.getvalue())

                    if uploaded_file.name in knowledge_base.documents_loaded:
                        status.info(
                            f"{uploaded_file.name} is already processed."
                        )
                    else:
                        status.info(
                            f"Processing {uploaded_file.name}..."
                        )

                        chunks_created = knowledge_base.add_document(
                            str(file_path)
                        )

                        if chunks_created > 0:
                            st.session_state.processed_files.append(
                                uploaded_file.name
                            )

                            status.success(
                                f"{uploaded_file.name}: "
                                f"{chunks_created} chunks created."
                            )
                        else:
                            status.warning(
                                f"No new chunks created for {uploaded_file.name}."
                            )

                except Exception as error:
                    status.error(
                        f"Error processing {uploaded_file.name}: {error}"
                    )

                progress.progress((index + 1) / total_files)

            st.session_state.processed_files = list(
                getattr(knowledge_base, "documents_loaded", [])
            )

            st.success("Document processing completed.")

    st.divider()

    st.subheader("Your Documents")

    if st.session_state.processed_files:
        for file_name in st.session_state.processed_files:
            st.markdown(f"📄 {file_name}")
    else:
        st.caption("No documents processed yet.")

    st.divider()

    st.metric(
        "Documents",
        len(st.session_state.processed_files)
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Local AI • Hugging Face • FAISS")


# ---------------------------------
# MAIN DASHBOARD
# ---------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Uploaded Documents",
    len(st.session_state.processed_files)
)

vector_chunks = len(
    getattr(knowledge_base.vector_store, "text_chunks", [])
)

col2.metric(
    "Knowledge Chunks",
    vector_chunks
)

col3.metric(
    "AI Model",
    "Qwen2.5 1.5B"
)

st.divider()


# ---------------------------------
# DOCUMENT SUMMARIZER
# ---------------------------------

st.header("📝 Document Summarizer")

st.write(
    "Generate a structured summary of any document in your Knowledge Base."
)

if not st.session_state.processed_files:

    st.info("Upload and process a document first to generate its summary.")

else:

    selected_document = st.selectbox(
        "Select a document to summarize",
        options=st.session_state.processed_files,
        key="summary_document_select"
    )

    if st.button("✨ Generate Summary", key="generate_summary"):

        document_chunks = get_document_chunks(
            selected_document,
            knowledge_base
        )

        if not document_chunks:
            st.warning(
                "No text chunks found for this document. "
                "Please process the document again."
            )

        else:
            try:
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
                st.error(f"Summary error: {error}")

    if (
        st.session_state.document_summary
        and st.session_state.summary_source == selected_document
    ):

        st.success("Summary generated successfully.")

        st.markdown("### 📑 Document Summary")

        st.markdown(st.session_state.document_summary)

        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.document_summary,
            file_name="document_summary.txt",
            mime="text/plain"
        )


st.divider()


# ---------------------------------
# QUIZ GENERATOR
# ---------------------------------

st.header("🎯 Quiz Generator")

st.write(
    "Create multiple-choice questions from your uploaded documents "
    "and test your understanding."
)

if not st.session_state.processed_files:

    st.info("Process a document first to generate a quiz.")

else:

    quiz_document = st.selectbox(
        "Choose a document for the quiz",
        options=st.session_state.processed_files,
        key="quiz_document_select"
    )

    number_of_questions = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button("🚀 Generate Quiz"):

        quiz_chunks = get_document_chunks(
            quiz_document,
            knowledge_base
        )

        if not quiz_chunks:
            st.warning("No text found for this document.")

        else:
            quiz_context = "\n\n".join(quiz_chunks)
            quiz_context = quiz_context[:6000]

            try:
                with st.spinner("Generating your quiz..."):

                    assistant = load_ai_assistant()

                    raw_quiz = generate_quiz(
                        quiz_context,
                        number_of_questions,
                        assistant
                    )

                    questions = parse_quiz(raw_quiz)

                    if questions:
                        st.session_state.quiz_questions = questions
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_source = quiz_document
                    else:
                        st.error(
                            "The AI did not return valid quiz questions. "
                            "Please try generating the quiz again."
                        )

            except Exception as error:
                st.error(f"Quiz generation failed: {error}")

    if (
        st.session_state.quiz_questions
        and st.session_state.quiz_source == quiz_document
    ):

        st.markdown("### 📚 Your Quiz")

        with st.form("quiz_form"):

            for index, question in enumerate(
                st.session_state.quiz_questions
            ):

                st.markdown(
                    f"**Q{index + 1}. {question['question']}**"
                )

                options = question["options"]

                selected_option = st.radio(
                    "Select your answer:",
                    options=["A", "B", "C", "D"],
                    format_func=lambda key, opts=options: (
                        f"{key}. {opts[key]}"
                    ),
                    key=f"quiz_question_{index}",
                    index=None
                )

                st.session_state.quiz_answers[index] = selected_option

                st.markdown("---")

            submitted = st.form_submit_button("Submit Quiz")

        if submitted:
            st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:

            score = 0
            total = len(st.session_state.quiz_questions)

            st.markdown("### 📊 Quiz Results")

            for index, question in enumerate(
                st.session_state.quiz_questions
            ):

                user_answer = st.session_state.quiz_answers.get(index)
                correct_answer = question["answer"]

                if user_answer == correct_answer:
                    score += 1
                    st.success(
                        f"Question {index + 1}: Correct!"
                    )
                else:
                    st.error(
                        f"Question {index + 1}: Incorrect. "
                        f"Correct answer: {correct_answer}. "
                        f"{question['options'][correct_answer]}"
                    )

                if question.get("explanation"):
                    st.caption(question["explanation"])

            percentage = (score / total) * 100 if total else 0

            st.metric(
                "Your Score",
                f"{score} / {total}"
            )

            st.progress(percentage / 100)

            st.write(f"**Percentage: {percentage:.1f}%**")

            if percentage >= 80:
                st.success("Excellent work! 🎉")
            elif percentage >= 50:
                st.info("Good effort! Keep practicing.")
            else:
                st.warning(
                    "Keep studying the document and try again."
                )

            if st.button("🔄 Try Another Quiz"):
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_source = ""
                st.rerun()


st.divider()


# ---------------------------------
# ASK YOUR DOCUMENTS - RAG CHAT
# ---------------------------------

st.header("💬 Ask Your Documents")

st.write(
    "Ask questions about your uploaded documents. "
    "The assistant searches your knowledge base before generating an answer."
)

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")


question = st.chat_input(
    "Ask something about your documents..."
)

if question:

    if not st.session_state.processed_files:

        st.warning("Please upload and process a document first.")

    else:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                with st.spinner("Searching your documents..."):

                    results = knowledge_base.search_documents(
                        question,
                        top_k=5
                    )

                if not results:

                    answer = (
                        "I could not find this information "
                        "in your uploaded documents."
                    )

                    sources = []

                else:

                    context = "\n\n".join(
                        item["text"]
                        for item in results
                        if item.get("text")
                    )

                    sources = list(dict.fromkeys(
                        item.get("source", "Unknown document")
                        for item in results
                    ))

                    if not context.strip():

                        answer = (
                            "I could not find readable information "
                            "for this question."
                        )

                    else:

                        with st.spinner("Generating answer..."):

                            assistant = load_ai_assistant()

                            answer = assistant.generate_answer(
                                question,
                                context
                            )

                            if not answer:
                                answer = (
                                    "I could not generate an answer. "
                                    "Please try again."
                                )

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.write(f"• {source}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as error:

                answer = f"An error occurred: {error}"

                st.error(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": []
                })


st.divider()


# ---------------------------------
# ABOUT
# ---------------------------------

with st.expander("ℹ️ About This Project"):

    st.markdown("""
    ### Personal Knowledge Assistant

    This application allows users to upload personal documents,
    search their content, generate summaries, and create quizzes.

    **Technologies Used:**

    - Python
    - Streamlit
    - Hugging Face Transformers
    - Qwen2.5-1.5B-Instruct local language model
    - Sentence Transformers
    - FAISS Vector Database
    - RAG (Retrieval-Augmented Generation)
    - PyPDF

    **How it works:**

    1. Upload a PDF or TXT document.
    2. Extract text from the document.
    3. Split text into smaller chunks.
    4. Convert chunks into embeddings.
    5. Store embeddings in FAISS.
    6. Retrieve relevant chunks for a question.
    7. Generate an answer using the local language model.
    """)


# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown(
    "<p class='small-note' style='text-align:center;'>"
    "Built with Python, Streamlit, Hugging Face and FAISS."
    "</p>",
    unsafe_allow_html=True
)