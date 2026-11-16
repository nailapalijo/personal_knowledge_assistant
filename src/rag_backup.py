
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_INPUT_TOKENS = 3000

NO_ANSWER = "I could not find this information in your documents."


class RAGAssistant:

    def __init__(self):

        print("Loading Qwen2.5-1.5B-Instruct...")

        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        )

        self.model.to(self.device)
        self.model.eval()

        print("Qwen model loaded successfully!")

    def _generate(self, messages, max_new_tokens=180):

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS
        ).to(self.device)

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[
            0, inputs["input_ids"].shape[1]:
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

    def generate_answer(self, question, context):

        if not question or not question.strip():
            return "Please enter a question."

        if not context or not context.strip():
            return NO_ANSWER

        system_prompt = """
You are a careful Personal Knowledge Assistant.

Answer questions using ONLY the supplied document context.

Rules:
1. Give accurate and relevant answers.
2. Never invent facts.
3. Do not use outside knowledge.
4. If the answer is missing from the context,
   say that you could not find it in the documents.
5. Keep answers clear and easy to understand.
6. Include important details when available.
7. If the context is insufficient, clearly mention that.
"""

        user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer using only the document context.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        answer = self._generate(
            messages,
            max_new_tokens=220
        )

        if not answer:
            return NO_ANSWER

        return answer

    def generate_content(
        self,
        instruction,
        context,
        max_new_tokens=300
    ):

        if not instruction or not instruction.strip():
            return "Please provide an instruction."

        if not context or not context.strip():
            return "No document context is available."

        system_prompt = """
You are a document-based AI assistant.

Follow the user's instructions carefully.
Use only the supplied document.
Do not invent facts or details.
If information is missing, state that clearly.
"""

        user_prompt = f"""
DOCUMENT CONTEXT:

{context}

TASK:

{instruction}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        result = self._generate(
            messages,
            max_new_tokens=max_new_tokens
        )

        if not result:
            return "The AI could not generate content. Please try again."

        return result
