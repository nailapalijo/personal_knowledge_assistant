import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"


class RAGAssistant:

    def __init__(self):
        print("Loading FLAN-T5 model from Hugging Face...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        self.model.eval()

        print("FLAN-T5 model loaded successfully!")

    def generate_answer(self, question, context):

        prompt = f"""
You are a document question answering assistant.

Answer using only the information in the context.
Do not guess or invent information.
Give a short and direct answer.

If the information is missing, say:
I could not find this information in your documents.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
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
                repetition_penalty=1.2
            )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        if not answer:
            return "I could not find this information in your documents."

        return answer