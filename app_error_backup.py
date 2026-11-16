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
))