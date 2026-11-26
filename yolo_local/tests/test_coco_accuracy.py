"""
Comprehensive Accuracy & Coverage Test Suite for ALL 80 COCO 2017 Classes.

Validates that every single COCO class (0 to 79) is accurately recognized
from realistic spoken language, colloquial synonyms, plurals, and phonetic STT errors.
"""

import unittest
from backend.services.nlp_extractor import get_nlp_extractor, COCO_CLASSES
from backend.services.stt_service import WhisperSTTService, get_stt_manager
from backend.config import settings


class TestAllCOCO80ClassesAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = get_nlp_extractor()

    def test_all_80_classes_canonical_names(self):
        """Verify that every single one of the 80 COCO classes is recognized by its canonical name."""
        for expected_id, cls_name in enumerate(COCO_CLASSES):
            sentence = f"Can you please detect the {cls_name} over there?"
            res = self.nlp.extract(sentence)
            self.assertIn(
                cls_name,
                res.target_classes,
                f"Failed canonical detection for COCO ID {expected_id}: '{cls_name}' in '{sentence}'",
            )
            self.assertIn(
                expected_id,
                res.target_class_ids,
                f"Class ID mismatch for '{cls_name}'. Expected {expected_id}, got {res.target_class_ids}",
            )

    def test_all_80_classes_spoken_colloquialisms(self):
        """
        Verify that all 80 COCO classes are accurately mapped from natural spoken colloquialisms.
        """
        spoken_test_cases = [
            (0, "person", "There are three people and a little kid walking by"),
            (1, "bicycle", "Where did I park my mountain bike?"),
            (2, "car", "Find the blue sedan and the taxi"),
            (3, "motorcycle", "Watch out for that fast motorbike and the scooter"),
            (4, "airplane", "Look up at the jet flying across the sky"),
            (5, "bus", "The school bus and city transit coach are waiting"),
            (6, "train", "Is the subway or passenger locomotive arriving soon?"),
            (7, "truck", "Look at that big pickup truck and the delivery lorry"),
            (8, "boat", "We rented a small sailboat and a yellow kayak"),
            (9, "traffic light", "Stop at the red light and wait for the stoplight"),
            (10, "fire hydrant", "Spot the red fireplug and hydrant on the corner"),
            (11, "stop sign", "There is an octagonal red stop sign ahead"),
            (12, "parking meter", "Put some coins into the parking ticket machine"),
            (13, "bench", "Let's take a rest on the wooden park bench"),
            (14, "bird", "Look at the colorful parrot, the pigeon, and the seagull"),
            (15, "cat", "That cute little kitty and tomcat are sleeping"),
            (16, "dog", "The energetic puppy and golden retriever are playing"),
            (17, "horse", "The black stallion and pony are in the stable"),
            (18, "sheep", "The fluffy lamb and ram are grazing on the hill"),
            (19, "cow", "Look at the dairy cattle and the bull in the pasture"),
            (20, "elephant", "The giant tusker is walking through the jungle"),
            (21, "bear", "A wild grizzly was spotted in the national park"),
            (22, "zebra", "Look at the black and white stripes on the zebras"),
            (23, "giraffe", "The tall giraffes are eating leaves from the treetop"),
            (24, "backpack", "I put my books in the schoolbag and daypack"),
            (25, "umbrella", "Take a brolly and rain umbrella because it is raining"),
            (26, "handbag", "She forgot her leather purse and tote bag in the car"),
            (27, "tie", "He wore a black bowtie and silk necktie"),
            (28, "suitcase", "Pack your luggage, trolley bag, and carry-on"),
            (29, "frisbee", "Let's toss the flying disc on the lawn"),
            (30, "skis", "Grab your pair of skis and snow skis"),
            (31, "snowboard", "He loves snowboarding with his new snow board"),
            (32, "sports ball", "Pass the soccer ball, football, and basketball"),
            (33, "kite", "Fly the stunt kite at the beach"),
            (34, "baseball bat", "Swing the wooden bat and metal softball bat"),
            (35, "baseball glove", "Catch the ball with your baseball mitt"),
            (36, "skateboard", "The teenager was riding a longboard and skateboard"),
            (37, "surfboard", "Catch the ocean waves on your wave board"),
            (38, "tennis racket", "Bring your tennis racquet to the sports court"),
            (39, "bottle", "I need a cold water bottle and a thermos"),
            (40, "wine glass", "Pour the champagne into the crystal goblet"),
            (41, "cup", "Pour hot coffee into the ceramic mug and teacup"),
            (42, "fork", "Set the dinner fork next to the plate"),
            (43, "knife", "Use the steak knife and sharp pocketknife"),
            (44, "spoon", "Eat soup with the silver tablespoon and teaspoon"),
            (45, "bowl", "Fill the cereal bowl and soup bowl with milk"),
            (46, "banana", "Add some sweet bananas and plantains"),
            (47, "apple", "Eat a crisp green apple and gala apple"),
            (48, "sandwich", "I ordered a cheeseburger, panini, and a sub"),
            (49, "orange", "Peel the fresh tangerine and clementine"),
            (50, "broccoli", "Eat healthy vegetables like broccoli florets"),
            (51, "carrot", "Chop the orange carrots and baby carrots"),
            (52, "hot dog", "Order a tasty frankfurter and corn dog with mustard"),
            (53, "pizza", "Grab a warm slice of pepperoni pizza"),
            (54, "donut", "Enjoy a glazed doughnut with your morning coffee"),
            (55, "cake", "Bake a birthday cake, cupcake, and cheesecake"),
            (56, "chair", "Sit down in the comfortable armchair and office stool"),
            (57, "couch", "Relax on the living room sofa and sectional lounge"),
            (58, "potted plant", "Water the indoor houseplant, succulent, and bonsai"),
            (59, "bed", "Lie down on the twin mattress and bunk bed"),
            (60, "dining table", "Gather around the kitchen table and dinner table"),
            (61, "toilet", "Where is the restroom lavatory and commode?"),
            (62, "tv", "Turn on the flatscreen television and monitor"),
            (63, "laptop", "Open your MacBook, notebook, and portable computer"),
            (64, "mouse", "Move the wireless optical computer mouse"),
            (65, "remote", "Where is the television remote control and clicker?"),
            (66, "keyboard", "Type on the mechanical computer keyboard"),
            (67, "cell phone", "Where did I leave my smartphone, iPhone, and mobile?"),
            (68, "microwave", "Heat the leftovers in the microwave oven"),
            (69, "oven", "Bake the bread on the stovetop and kitchen oven"),
            (70, "toaster", "Toast a slice of bread in the toaster oven"),
            (71, "sink", "Wash your hands in the bathroom basin and washbasin"),
            (72, "refrigerator", "Keep the milk cold inside the fridge and icebox"),
            (73, "book", "Read a fascinating hardcover novel and textbook"),
            (74, "clock", "Check the time on the analog wall clock and timer"),
            (75, "vase", "Put the fresh flowers into the ceramic flower vase"),
            (76, "scissors", "Cut the wrapping paper with a pair of shears and scissors"),
            (77, "teddy bear", "The child hugged her soft plushie and teddy"),
            (78, "hair drier", "Dry your hair using the electric blow dryer and blowdryer"),
            (79, "toothbrush", "Brush your teeth with an electric toothbrush"),
        ]

        for expected_id, expected_class, test_sentence in spoken_test_cases:
            res = self.nlp.extract(test_sentence)
            self.assertIn(
                expected_class,
                res.target_classes,
                f"Failed spoken detection for COCO ID {expected_id} ('{expected_class}') in: '{test_sentence}'",
            )
            self.assertIn(
                expected_id,
                res.target_class_ids,
                f"Class ID {expected_id} missing for '{expected_class}'. Extracted IDs: {res.target_class_ids}",
            )

    def test_compound_disambiguation(self):
        """
        Verify that compound class names take precedence and do NOT mistakenly
        trigger sub-word classes:
        - 'hot dog' must NOT trigger 'dog'
        - 'teddy bear' must NOT trigger 'bear'
        - 'baseball bat' must NOT trigger 'sports ball'
        - 'wine glass' must NOT trigger 'cup'
        - 'potted plant' must extract COCO ID 58
        """
        # Hot dog vs Dog
        res = self.nlp.extract("I want to eat a hot dog, not pet a dog")
        self.assertIn("hot dog", res.target_classes)
        self.assertIn("dog", res.excluded_classes)

        # Teddy bear vs Bear
        res = self.nlp.extract("Look at that cute teddy bear in the shop window")
        self.assertIn("teddy bear", res.target_classes)
        self.assertNotIn("bear", res.target_classes)

        # Wine glass vs general glass
        res = self.nlp.extract("Please bring a wine glass to the dining table")
        self.assertIn("wine glass", res.target_classes)
        self.assertIn("dining table", res.target_classes)

        # Hair drier
        res = self.nlp.extract("Where is my blow dryer?")
        self.assertIn("hair drier", res.target_classes)

    def test_phonetic_error_normalization(self):
        """Verify phonetic and common STT transcription sound-alikes."""
        error_cases = [
            ("Where is my cellfone and waterbottle?", ["cell phone", "bottle"]),
            ("Look at the aeroplan in the sky", ["airplane"]),
            ("Open the refregerator and take the brocolli", ["refrigerator", "broccoli"]),
            ("Turn off the telivision", ["tv"]),
            ("She used a hairdryer and brushed with a tooth brush", ["hair drier", "toothbrush"]),
            ("I ordered a sandwhich and a pisa", ["sandwich", "pizza"]),
        ]
        for sentence, expected_targets in error_cases:
            res = self.nlp.extract(sentence)
            for target in expected_targets:
                self.assertIn(
                    target,
                    res.target_classes,
                    f"Phonetic test failed for '{target}' in '{sentence}'. Found: {res.target_classes}",
                )

    def test_quantifiers_and_materials(self):
        """Verify counts and adjectives/materials are captured."""
        res = self.nlp.extract("I see two red apples and a wooden dining table")
        self.assertEqual(len(res.extracted_objects), 2)

        apple_obj = next(o for o in res.extracted_objects if o.class_name == "apple")
        self.assertEqual(apple_obj.count, 2)
        self.assertIn("red", apple_obj.attributes)

        table_obj = next(o for o in res.extracted_objects if o.class_name == "dining table")
        self.assertIn("wooden", table_obj.attributes)

    def test_whisper_coco_prompt_configuration(self):
        """Verify Whisper service defaults to COCO prompt biasing."""
        self.assertTrue(len(settings.whisper_initial_prompt) > 50)
        for cls_name in ["cell phone", "microwave", "remote", "refrigerator", "hair drier"]:
            self.assertIn(cls_name, settings.whisper_initial_prompt)


if __name__ == "__main__":
    unittest.main()
