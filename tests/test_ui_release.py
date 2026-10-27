import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class UIReleaseRegressionTests(unittest.TestCase):
    def test_theme_styles_are_installed_before_local_dev_banner(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertLess(app.index("<style>"), app.index("Local development mode"))

    def test_local_dev_banner_has_explicit_high_contrast_text(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn(".dev-mode-notice, .dev-mode-notice *", app)
        self.assertIn("-webkit-text-fill-color:#273449 !important", app)

    def test_primary_buttons_use_steel_gray_and_white_text(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("background:#64748b !important", app)
        self.assertIn("-webkit-text-fill-color:#ffffff !important", app)

    def test_no_green_primary_theme_remains(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        config = (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")
        self.assertNotIn("#13b86b", app + config)
        self.assertIn('primaryColor="#64748b"', config)

if __name__ == "__main__":
    unittest.main()
