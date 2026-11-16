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

                answer_audio = None
                if speak_answers and answer and not answer.startswith("An error occurred:"):
                    try:
                        with st.spinner("Preparing voice response..."):
                            answer_audio = create_speech_audio(
                                answer,
                                selected_tts_voice
                            )
                    except Exception as speech_error:
                        st.warning(
                            "Text answer is ready, but voice generation failed. "
                            f"Check your internet connection. Details: {speech_error}"
                        )

                st.markdown(answer)

                if answer_audio:
                    st.audio(answer_audio, format="audio/mp3")

                if sources:

                    with st.expander("📚 Sources"):

                        for source in sources:
                            st.write(f"• {source}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "audio": answer_audio
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
    - Qwen2.5-0.5B-Instruct local language model
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

    Quiz questions are generated locally from document text.
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
