import json
import math
import re
from pathlib import Path


_SUMMARY_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "will", "with"
}
_QUIZ_STOP_WORDS = _SUMMARY_STOP_WORDS | {
    "what", "which", "who", "where", "when", "why", "how", "does",
    "do", "did", "can", "could", "would", "should", "according"
}
_ANSWER_LABEL = re.compile(r"^\s*([A-D])(?:[.)]\s*|\s+)", re.IGNORECASE)


def _words(text):
    return re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)


def _meaningful_words(text, stop_words):
    return [
        word for word in _words(text)
        if len(word) > 1 and word not in stop_words
    ]


def get_document_chunks(document_name, knowledge_base):
    vector_store = knowledge_base.vector_store
    text_chunks = vector_store.text_chunks
    sources = vector_store.sources

    if len(text_chunks) != len(sources):
        raise ValueError("The document index has inconsistent source metadata.")

    expected_name = Path(document_name).name.casefold()
    return [
        str(chunk)
        for chunk, source in zip(text_chunks, sources)
        if Path(str(source)).name.casefold() == expected_name and str(chunk).strip()
    ]


def summarize_document(document_name, knowledge_base, sentence_limit=7, word_limit=150):
    chunks = get_document_chunks(document_name, knowledge_base)
    if not chunks:
        return "No extracted text is available for this document."

    sentences = []
    seen = set()
    for chunk in chunks:
        for candidate in re.split(r"(?<=[.!?])\s+|[\r\n]+", chunk):
            sentence = re.sub(r"\s+", " ", candidate).strip(" \t-•")
            words = _meaningful_words(sentence, _SUMMARY_STOP_WORDS)
            normalized = " ".join(_words(sentence))
            if len(words) < 3 or normalized in seen:
                continue
            seen.add(normalized)
            sentences.append((sentence, words))

    if not sentences:
        return "No readable summary text is available for this document."

    document_frequency = {}
    for _, words in sentences:
        for word in set(words):
            document_frequency[word] = document_frequency.get(word, 0) + 1

    sentence_scores = []
    for index, (_, words) in enumerate(sentences):
        unique_words = set(words)
        score = sum(
            math.log((len(sentences) + 1) / (document_frequency[word] + 1)) + 1
            for word in unique_words
        ) / math.sqrt(len(words))
        sentence_scores.append((score, index))

    selected = []
    selected_words = 0
    for _, index in sorted(sentence_scores, reverse=True):
        sentence = sentences[index][0]
        sentence_word_count = len(_words(sentence))
        if selected and selected_words + sentence_word_count > word_limit:
            continue
        selected.append(index)
        selected_words += sentence_word_count
        if len(selected) >= sentence_limit:
            break

    return "\n\n".join(sentences[index][0] for index in sorted(selected))


def parse_quiz(text):
    if not isinstance(text, str) or not text.strip():
        return []

    candidate = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE
    )

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("[")
        if start < 0:
            return []

        data = []
        object_start = None
        object_depth = 0
        in_string = False
        escaped = False

        for index in range(start + 1, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                if object_depth == 0:
                    object_start = index
                object_depth += 1
            elif char == "}":
                object_depth -= 1
                if object_depth == 0 and object_start is not None:
                    try:
                        data.append(json.loads(candidate[object_start:index + 1]))
                    except json.JSONDecodeError:
                        pass
                    object_start = None

        if not data:
            return []

    return data if isinstance(data, list) else []


def _clean_option(option):
    option = str(option).strip()
    return _ANSWER_LABEL.sub("", option, count=1).strip()


def _supported_by_source(candidate, source_words):
    words = _meaningful_words(candidate, _QUIZ_STOP_WORDS)
    return bool(words) and all(word in source_words for word in words)


def validate_quiz(data, context):
    if not isinstance(data, list):
        return []

    source_words = set(_words(context))
    validated = []
    seen_questions = set()

    for item in data:
        if not isinstance(item, dict):
            continue

        question = str(item.get("question", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        difficulty = str(item.get("difficulty", "")).strip().capitalize()
        options = item.get("options")
        if (
            not question
            or not explanation
            or difficulty not in {"Easy", "Medium", "Hard"}
            or not isinstance(options, list)
            or not 3 <= len(options) <= 4
        ):
            continue

        cleaned_options = [_clean_option(option) for option in options]
        normalized_options = [" ".join(_words(option)) for option in cleaned_options]
        if (
            any(not option for option in cleaned_options)
            or len(set(normalized_options)) != len(normalized_options)
        ):
            continue

        answer = str(item.get("answer", "")).strip()
        answer_match = re.match(r"^\s*([A-D])(?:[.)]|\s|$)", answer, re.IGNORECASE)
        if answer_match:
            answer_index = ord(answer_match.group(1).upper()) - ord("A")
        else:
            normalized_answer = " ".join(_words(_clean_option(answer)))
            matching_options = [
                index for index, option in enumerate(normalized_options)
                if option == normalized_answer
            ]
            if len(matching_options) != 1:
                continue
            answer_index = matching_options[0]

        if len(cleaned_options) == 3:
            if answer_match and answer_index >= len(cleaned_options):
                continue
            cleaned_options.append("Not stated in the document")
            normalized_options.append(" ".join(_words(cleaned_options[-1])))
            if len(set(normalized_options)) != 4:
                continue

        correct_option = cleaned_options[answer_index]
        if not _supported_by_source(correct_option, source_words):
            continue

        question_words = set(_meaningful_words(question, _QUIZ_STOP_WORDS))
        if not question_words.intersection(source_words):
            continue

        explanation_words = set(_meaningful_words(explanation, _QUIZ_STOP_WORDS))
        answer_words = set(_meaningful_words(correct_option, _QUIZ_STOP_WORDS))
        if (
            not explanation_words.intersection(source_words)
            or not answer_words.intersection(explanation_words)
        ):
            continue

        normalized_question = " ".join(_words(question))
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)

        validated.append({
            "question": question,
            "options": [
                f"{chr(ord('A') + index)}. {option}"
                for index, option in enumerate(cleaned_options)
            ],
            "answer": chr(ord("A") + answer_index),
            "explanation": explanation,
            "difficulty": difficulty
        })

        if len(validated) == 5:
            break

    return validated


def generate_quiz(document_name, knowledge_base, assistant):
    chunks = get_document_chunks(document_name, knowledge_base)
    if not chunks:
        return []

    if len(chunks) > 8:
        selected_indices = [
            round(index * (len(chunks) - 1) / 7)
            for index in range(8)
        ]
        chunks = [chunks[index] for index in selected_indices]

    context = "\n\n".join(chunks)
    instruction = """
Create five concise multiple-choice questions based only on the supplied
document. Return a JSON array with no surrounding text and stop immediately
after the closing bracket. Keep each explanation to one short sentence. Every
item must contain:
"question", "options" (exactly four distinct answer strings), "answer"
(the correct option letter A, B, C, or D), "explanation" (supported by the
document and repeating the correct answer), and "difficulty" (Easy, Medium,
or Hard). There must be exactly one correct option per question. Use only
facts explicitly stated in the document; do not make up distractor facts.
If the document does not support a question, do not include it.
"""
    generated = assistant.generate_content(
        instruction,
        context,
        max_new_tokens=450
    )
    return validate_quiz(parse_quiz(generated), context)
