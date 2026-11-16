import tempfile
from pathlib import Path


def transcribe_audio(audio_bytes, language, model):
    audio_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            audio_path = temp_audio.name

        segments, info = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False
        )
        transcript = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()
        return transcript, info.language
    finally:
        if audio_path and Path(audio_path).exists():
            Path(audio_path).unlink()
