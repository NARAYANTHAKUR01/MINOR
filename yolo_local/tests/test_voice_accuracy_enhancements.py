"""
Unit tests validating voice object name accuracy enhancements for Omii.

Covers:
1. Single-word object name extraction.
2. Natural one-shot wake word + command combinations ("Hi Omii find my bottle").
3. Everyday colloquial objects (bag, glass, glasses, plate, dish, TV, phone).
4. Phonetic & STT sound-alike corrections (lap atop, study bear, tee vee, cub, bottl).
5. False-positive resistance (slippers does not match scissors or bottle).
6. Smart client hint fusion in interactive assistant audio processing.
"""

import unittest
from backend.services.nlp_extractor import get_nlp_extractor
from backend.services.interactive_assistant import get_interactive_assistant
from backend.config import settings


class TestVoiceAccuracyEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = get_nlp_extractor()
        cls.assistant = get_interactive_assistant()

    def test_single_word_and_colloquial_object_names(self):
        """Verify everyday colloquial object names map accurately to COCO classes."""
        test_cases = [
            ("bottle", "bottle"),
            ("cup", "cup"),
            ("mug", "cup"),
            ("glass", "wine glass"),
            ("glasses", "wine glass"),
            ("drinking glass", "wine glass"),
            ("water glass", "wine glass"),
            ("water bottle", "bottle"),
            ("bag", "backpack"),
            ("my bag", "backpack"),
            ("backpack", "backpack"),
            ("purse", "handbag"),
            ("wallet", "handbag"),
            ("plate", "bowl"),
            ("dish", "bowl"),
            ("phone", "cell phone"),
            ("cellphone", "cell phone"),
            ("mobile", "cell phone"),
            ("smart phone", "cell phone"),
            ("laptop", "laptop"),
            ("computer", "laptop"),
            ("macbook", "laptop"),
            ("pc", "laptop"),
            ("tv", "tv"),
            ("t.v.", "tv"),
            ("teevee", "tv"),
            ("television", "tv"),
            ("couch", "couch"),
            ("sofa", "couch"),
            ("table", "dining table"),
            ("desk", "dining table"),
            ("chair", "chair"),
            ("plant", "potted plant"),
            ("remote", "remote"),
            ("bed", "bed"),
            ("scissors", "scissors"),
            ("car", "car"),
        ]

        for query, expected_class in test_cases:
            res = self.nlp.extract_target_and_intent(query)
            self.assertTrue(
                res["valid"],
                f"Failed to extract valid target for: '{query}'",
            )
            self.assertEqual(
                res["target_object"],
                expected_class,
                f"Expected '{expected_class}' for query '{query}', got '{res['target_object']}'",
            )

    def test_one_shot_wake_command_phrases(self):
        """Verify queries with leading wake phrases cleanly extract the core object name."""
        cases = [
            ("Hi Omii, where is my phone?", "cell phone"),
            ("Hey Omii find the bottle", "bottle"),
            ("Hi Omii bottle", "bottle"),
            ("Hello Omii, look for the cup", "cup"),
            ("Omii find my keys and phone", "cell phone"),
            ("Ok Omii, where is my bag?", "backpack"),
            ("Hi Omii laptop", "laptop"),
            ("Omii where is the TV", "tv"),
        ]
        for query, expected_target in cases:
            res = self.nlp.extract_target_and_intent(query)
            self.assertTrue(res["valid"], f"Failed on one-shot query: '{query}'")
            self.assertEqual(
                res["target_object"],
                expected_target,
                f"Expected '{expected_target}' from '{query}', got '{res['target_object']}'",
            )

    def test_phonetic_and_stt_soundalikes(self):
        """Verify STT phonetic errors are corrected to their proper object classes."""
        cases = [
            ("lap atop", "laptop"),
            ("study bear", "teddy bear"),
            ("tee vee", "tv"),
            ("cub", "cup"),
            ("bottl", "bottle"),
            ("sandwhich", "sandwich"),
            ("cheir", "chair"),
            ("moniter", "tv"),
            ("bagpack", "backpack"),
        ]
        for query, expected in cases:
            res = self.nlp.extract_target_and_intent(query)
            self.assertTrue(res["valid"], f"Failed phonetic case: '{query}'")
            self.assertEqual(
                res["target_object"],
                expected,
                f"Expected '{expected}' for '{query}', got '{res['target_object']}'",
            )

    def test_false_positive_guardrails(self):
        """Verify fuzzy matching does not trigger false positives on unrelated words."""
        negative_words = ["slippers", "computerized", "reminiscent", "unbelievable"]
        for word in negative_words:
            res = self.nlp.extract_target_and_intent(word)
            self.assertFalse(
                res["valid"],
                f"False positive detected: '{word}' was mistakenly identified as '{res['target_object']}'",
            )

    def test_whisper_configuration_language(self):
        """Verify whisper settings explicitly enforce English for speed and accuracy."""
        self.assertEqual(settings.whisper_language, "en")
        self.assertIn("bottle", settings.whisper_initial_prompt)
        self.assertIn("cup", settings.whisper_initial_prompt)
        self.assertIn("cell phone", settings.whisper_initial_prompt)

    def test_assistant_process_audio_with_hint_fallback(self):
        """Verify process_audio_command recovers target using hint_text when audio is empty/unclear."""
        self.assistant.enable()
        # Empty audio bytes, but client hint provided
        resp = self.assistant.process_audio_command(
            audio_bytes=b"",
            hint_text="where is my water bottle",
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.target_object, "bottle")
        self.assistant.disable()


if __name__ == "__main__":
    unittest.main()
