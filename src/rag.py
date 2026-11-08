
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "models/flan-t5-small"


class RAGAssistant:

    def __init__(self):

        print("Loading local FLAN-T5 model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            use_fast=False
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            use_safetensors=True
        )

        self.model.eval()

        print("FLAN-T5 model loaded successfully!")

    def generate_answer(self, question, context):

        prompt = f"""
You are a document question answering assistant.

Read the CONTEXT carefully and answer the QUESTION using only information explicitly stated in the context.

Rules:
1. Give a short, direct answer.
2. Do not guess or invent information.
3. Do not mix unrelated details.
4. Do not treat an office, training center, or internship location as a university.
5. If the answer is not clearly available, say exactly:
I could not find this information in your documents.

CONTEXT:
{context}

QUESTION:
{question}

DIRECT ANSWER:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.2,
                early_stopping=True
            )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        answer = answer.strip()

        if not answer or answer.lower() in ["a).", "a)", "answer:"]:
            return "I could not find this information in your documents."

        return answer
