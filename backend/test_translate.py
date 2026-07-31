import unittest

from app import services


class TranslateServiceTests(unittest.TestCase):
    def test_translate_french_does_not_return_placeholder(self):
        text = "Welcome to Voice Reader"
        result = services.translate(text, "French")
        self.assertIn("Bienvenue", result)
        self.assertNotIn("translation service not configured", result)


if __name__ == "__main__":
    unittest.main()
