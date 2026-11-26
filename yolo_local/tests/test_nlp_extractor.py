"""
Comprehensive Automated Test Suite for NLP Object Extraction Engine.
Runs with standard library unittest (no external test runner dependencies needed).

Covers:
- All 80 COCO classes verification
- Longest-phrase precedence (e.g. hot dog vs dog, teddy bear vs bear)
- Colloquial synonyms & spoken aliases
- Irregular and regular plural lemmatization
- Context-aware negation handling
- Intent classification
- Attribute and quantifier extraction
- Fuzzy matching tolerance for STT spelling noise
"""

import sys
import unittest
from pathlib import Path

# Ensure yolo_local root is in sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.nlp_extractor import (
    CLASS_NAME_TO_ID,
    COCO_CLASSES,
    NLPExtractor,
    get_nlp_extractor,
)


class TestNLPExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = get_nlp_extractor()

    def test_coco_classes_count_and_mapping(self):
        """Ensure exactly 80 COCO classes are registered in correct index order."""
        self.assertEqual(len(COCO_CLASSES), 80)
        self.assertEqual(self.extractor.coco_classes[0], "person")
        self.assertEqual(self.extractor.coco_classes[79], "toothbrush")
        self.assertEqual(CLASS_NAME_TO_ID["car"], 2)
        self.assertEqual(CLASS_NAME_TO_ID["bottle"], 39)
        self.assertEqual(CLASS_NAME_TO_ID["cell phone"], 67)
        self.assertEqual(CLASS_NAME_TO_ID["dining table"], 60)

    def test_all_80_classes_direct_extraction(self):
        """Verify that every single one of the 80 classes is cleanly extractable."""
        for idx, name in enumerate(COCO_CLASSES):
            sentence = f"Please detect the {name} in the room."
            res = self.extractor.extract(sentence)
            self.assertIn(
                name,
                res.target_classes,
                f"Failed to extract class '{name}' from '{sentence}'",
            )
            self.assertIn(
                idx,
                res.target_class_ids,
                f"Class ID mismatch for '{name}'",
            )

    def test_longest_phrase_precedence(self):
        """Ensure compound multi-word classes take precedence over sub-words."""
        # 'hot dog' must not trigger 'dog'
        res = self.extractor.extract("I ordered a hot dog for lunch.")
        self.assertIn("hot dog", res.target_classes)
        self.assertNotIn("dog", res.target_classes)

        # 'teddy bear' must not trigger 'bear'
        res = self.extractor.extract("The kid is hugging a teddy bear.")
        self.assertIn("teddy bear", res.target_classes)
        self.assertNotIn("bear", res.target_classes)

        # 'wine glass' vs 'cup'
        res = self.extractor.extract("Place the wine glass on the dining table.")
        self.assertIn("wine glass", res.target_classes)
        self.assertIn("dining table", res.target_classes)

        # 'baseball bat' and 'baseball glove'
        res = self.extractor.extract("He dropped the baseball bat and the baseball glove.")
        self.assertIn("baseball bat", res.target_classes)
        self.assertIn("baseball glove", res.target_classes)

    def test_spoken_aliases_and_colloquialisms(self):
        """Verify natural spoken vocabulary maps accurately to COCO classes."""
        cases = [
            ("Where did I leave my phone?", "cell phone"),
            ("I need to find my smartphone", "cell phone"),
            ("Look at that cute puppy", "dog"),
            ("The kitten is sleeping", "cat"),
            ("He rode his bike to school", "bicycle"),
            ("She is resting on the sofa", "couch"),
            ("Put the milk in the fridge", "refrigerator"),
            ("Turn on the television", "tv"),
            ("Open my macbook", "laptop"),
            ("I need a coffee mug", "cup"),
            ("A yellow cab stopped here", "car"),
            ("Check the airplane in the sky", "airplane"),
            ("Pick up your rucksack", "backpack"),
            ("Grab the umbrella because it rains", "umbrella"),
            ("Clean the washbasin", "sink"),
            ("She lost her purse", "handbag"),
            ("Throw the football", "sports ball"),
        ]
        for prompt, expected_class in cases:
            res = self.extractor.extract(prompt)
            self.assertIn(
                expected_class,
                res.target_classes,
                f"Expected '{expected_class}' for prompt '{prompt}'",
            )

    def test_lemmatization_and_irregular_plurals(self):
        """Verify regular and irregular plural forms are correctly reduced."""
        # Irregular plurals
        res1 = self.extractor.extract("Three people walked into the store.")
        self.assertIn("person", res1.target_classes)

        res2 = self.extractor.extract("The chef sharpened two knives.")
        self.assertIn("knife", res2.target_classes)

        res3 = self.extractor.extract("Watch out for mice under the sink.")
        self.assertIn("mouse", res3.target_classes)
        self.assertIn("sink", res3.target_classes)

        # Regular plurals (-s, -es, -ies)
        res4 = self.extractor.extract("I see four bicycles, three benches, and two sandwiches.")
        self.assertIn("bicycle", res4.target_classes)
        self.assertIn("bench", res4.target_classes)
        self.assertIn("sandwich", res4.target_classes)

        # Inherent plurals
        res5 = self.extractor.extract("Bring your skis and scissors.")
        self.assertIn("skis", res5.target_classes)
        self.assertIn("scissors", res5.target_classes)

    def test_negation_detection(self):
        """Verify negation filters out excluded classes while keeping desired targets."""
        # "not X, but Y"
        res1 = self.extractor.extract("Find the apple, but not the orange.")
        self.assertIn("apple", res1.target_classes)
        self.assertIn("orange", res1.excluded_classes)
        self.assertNotIn("orange", res1.target_classes)

        # "avoid X"
        res2 = self.extractor.extract("Locate the laptop and avoid the television.")
        self.assertIn("laptop", res2.target_classes)
        self.assertIn("tv", res2.excluded_classes)
        self.assertNotIn("tv", res2.target_classes)

        # "without X"
        res3 = self.extractor.extract("Show cars without any trucks.")
        self.assertIn("car", res3.target_classes)
        self.assertIn("truck", res3.excluded_classes)
        self.assertNotIn("truck", res3.target_classes)

    def test_intent_classification(self):
        """Verify conversational intent detection."""
        self.assertEqual(self.extractor.detect_intent("Where is my water bottle?"), "locate")
        self.assertEqual(self.extractor.detect_intent("Locate the keys and cell phone"), "locate")
        self.assertEqual(self.extractor.detect_intent("How many people are in the room?"), "count")
        self.assertEqual(self.extractor.detect_intent("Count the bicycles"), "count")
        self.assertEqual(self.extractor.detect_intent("Can you see a cat?"), "detect")
        self.assertEqual(self.extractor.detect_intent("Do you see an airplane?"), "detect")
        self.assertEqual(self.extractor.detect_intent("Only show the chairs"), "filter")
        self.assertEqual(self.extractor.detect_intent("There is a dog and a cat"), "general")

    def test_quantifiers_and_attributes(self):
        """Verify extraction of counts and adjective attributes."""
        res = self.extractor.extract("I see three people and two bicycles near the bus.")
        person_obj = next(o for o in res.extracted_objects if o.class_name == "person")
        bicycle_obj = next(o for o in res.extracted_objects if o.class_name == "bicycle")
        bus_obj = next(o for o in res.extracted_objects if o.class_name == "bus")

        self.assertEqual(person_obj.count, 3)
        self.assertEqual(bicycle_obj.count, 2)
        self.assertIsNone(bus_obj.count)

        res2 = self.extractor.extract("Pick the red apple and two green bottles.")
        bottle_obj = next(o for o in res2.extracted_objects if o.class_name == "bottle")
        self.assertEqual(bottle_obj.count, 2)
        self.assertIn("green", bottle_obj.attributes)

    def test_fuzzy_matching_stt_noise(self):
        """Verify tolerance for common Speech-To-Text phonetic/spelling noise."""
        res1 = self.extractor.extract("Look at the aeroplan flying high.")
        self.assertIn("airplane", res1.target_classes)

        res2 = self.extractor.extract("Did you see my cellfone?")
        self.assertIn("cell phone", res2.target_classes)

        res3 = self.extractor.extract("Turn off the telivision.")
        self.assertIn("tv", res3.target_classes)

        res4 = self.extractor.extract("Where is my toothebrush?")
        self.assertIn("toothbrush", res4.target_classes)


if __name__ == "__main__":
    unittest.main()
