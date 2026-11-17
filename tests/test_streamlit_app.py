import builtins
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTests(unittest.TestCase):
    def test_cloud_app_starts_without_importing_model_packages(self):
        real_import = builtins.__import__
        forbidden = (
            "torch",
            "transformers",
            "sentence_transformers",
            "faster_whisper",
        )

        def guarded_import(name, *args, **kwargs):
            if any(name == module or name.startswith(module + ".") for module in forbidden):
                raise AssertionError(f"Unexpected startup model import: {name}")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"PKA_MODE": "cloud"}):
            with patch("builtins.__import__", side_effect=guarded_import):
                app = AppTest.from_file(str(ROOT / "app.py")).run()

        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertEqual(len(app.title), 1)

    def test_mobile_styles_cover_breakpoints_and_touch_sized_controls(self):
        for entrypoint in (ROOT / "app.py", ROOT / "src" / "app.py"):
            source = entrypoint.read_text(encoding="utf-8")
            with self.subTest(entrypoint=entrypoint):
                self.assertIn("@media (max-width: 768px)", source)
                self.assertIn("@media (max-width: 480px)", source)
                self.assertIn("min-height: 44px", source)
                self.assertIn('[data-testid="stColumn"]', source)
                self.assertIn(":has(.stButton)", source)
                self.assertIn('[data-testid="stChatInputSubmitButton"]', source)
                self.assertIn("overflow-wrap: anywhere", source)
                self.assertIn('initial_sidebar_state="auto"', source)


if __name__ == "__main__":
    unittest.main()
