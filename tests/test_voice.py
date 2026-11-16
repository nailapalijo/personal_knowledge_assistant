import os
import unittest
from types import SimpleNamespace

from src.voice import transcribe_audio


class VoiceTranscriptionTests(unittest.TestCase):
    def test_transcription_uses_selected_language_and_cleans_temp_audio(self):
        class FakeModel:
            def transcribe(self, audio_path, **kwargs):
                self.audio_path = audio_path
                self.options = kwargs
                with open(audio_path, "rb") as audio_file:
                    self.audio = audio_file.read()
                return (
                    [SimpleNamespace(text="  What is my name?  ")],
                    SimpleNamespace(language="en"),
                )

        model = FakeModel()
        transcript, detected_language = transcribe_audio(b"wav-data", "en", model)

        self.assertEqual(transcript, "What is my name?")
        self.assertEqual(detected_language, "en")
        self.assertEqual(model.audio, b"wav-data")
        self.assertEqual(model.options["language"], "en")
        self.assertEqual(model.options["task"], "transcribe")
        self.assertFalse(os.path.exists(model.audio_path))

    def test_auto_detection_passes_no_forced_language(self):
        class FakeModel:
            def transcribe(self, audio_path, **kwargs):
                self.language = kwargs["language"]
                return [SimpleNamespace(text="Urdu transcript")], SimpleNamespace(language="ur")

        model = FakeModel()
        transcript, detected_language = transcribe_audio(b"wav-data", None, model)
        self.assertEqual(transcript, "Urdu transcript")
        self.assertEqual(model.language, None)
        self.assertEqual(detected_language, "ur")


if __name__ == "__main__":
    unittest.main()
