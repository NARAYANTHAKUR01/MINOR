"""
Unit tests for Anchor-Based Relative Spatial Reasoning.

Validates:
1. Centroid calculation: center_x = (x1 + x2)/2, center_y = (y1 + y2)/2.
2. Relative direction determination (left, right, above, below, compound diagonals).
3. Tolerance threshold dead-zone handling.
4. Anchor object selection (prioritizing confidence & proximity, excluding target class).
5. Single target with clear anchor description.
6. Multiple targets handling (clearest target reported with total count).
7. Fallback when no anchor is available.
8. Missing target handling.
"""

import unittest
from backend.schemas.detection import BoundingBox, DetectionItem
from backend.services.spatial_reasoner import (
    AnchorSelectionConfig,
    RelativeSpatialResult,
    SpatialReasoner,
    calculate_centroid,
    determine_relative_direction,
    select_anchor_object,
    analyze_relative_spatial_relations,
)


class TestSpatialCentroidCalculation(unittest.TestCase):
    """Test 2D bounding box centroid calculations."""

    def test_centroid_from_bounding_box(self):
        bbox = BoundingBox(x1=100.0, y1=200.0, x2=300.0, y2=400.0, width=200.0, height=200.0)
        cx, cy = calculate_centroid(bbox)
        # (100 + 300) / 2 = 200, (200 + 400) / 2 = 300
        self.assertEqual(cx, 200.0)
        self.assertEqual(cy, 300.0)

    def test_centroid_from_dict(self):
        bbox_dict = {"x1": 50.0, "y1": 80.0, "x2": 150.0, "y2": 120.0}
        cx, cy = calculate_centroid(bbox_dict)
        self.assertEqual(cx, 100.0)
        self.assertEqual(cy, 100.0)

    def test_centroid_zero_coordinate(self):
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=640.0, y2=480.0, width=640.0, height=480.0)
        cx, cy = calculate_centroid(bbox)
        self.assertEqual(cx, 320.0)
        self.assertEqual(cy, 240.0)


class TestRelativeDirectionComparison(unittest.TestCase):
    """Test direction comparison logic with configurable tolerances."""

    def test_pure_left_and_right(self):
        anchor_center = (300.0, 300.0)

        # Target significantly to the left (dx = -200, dy = 0)
        target_left = (100.0, 300.0)
        dir_left = determine_relative_direction(target_left, anchor_center, tolerance_x=20.0, tolerance_y=20.0)
        self.assertEqual(dir_left, "to the left of")

        # Target significantly to the right (dx = +200, dy = 0)
        target_right = (500.0, 300.0)
        dir_right = determine_relative_direction(target_right, anchor_center, tolerance_x=20.0, tolerance_y=20.0)
        self.assertEqual(dir_right, "to the right of")

    def test_pure_above_and_below(self):
        anchor_center = (300.0, 300.0)

        # Target above (in image coordinates, smaller y means higher up)
        target_above = (300.0, 100.0)
        dir_above = determine_relative_direction(target_above, anchor_center, tolerance_x=20.0, tolerance_y=20.0)
        self.assertEqual(dir_above, "above")

        # Target below (larger y)
        target_below = (300.0, 500.0)
        dir_below = determine_relative_direction(target_below, anchor_center, tolerance_x=20.0, tolerance_y=20.0)
        self.assertEqual(dir_below, "below")

    def test_compound_diagonals(self):
        anchor_center = (300.0, 300.0)

        # Above and to the left
        target_tl = (150.0, 150.0)
        dir_tl = determine_relative_direction(target_tl, anchor_center, tolerance_x=20.0, tolerance_y=20.0)
        self.assertEqual(dir_tl, "above and to the left of")

        # Below and to the right
        target_br = (450.0, 450.0)
        dir_br = determine_relative_direction(target_br, anchor_center, tolerance_x=20.0, tolerance_y=20.0)
        self.assertEqual(dir_br, "below and to the right of")

    def test_tolerance_deadzone(self):
        anchor_center = (300.0, 300.0)

        # Difference is 10px, which is within the 25px tolerance -> treated as vertically aligned
        target_slight_y = (100.0, 310.0)
        dir_slight = determine_relative_direction(target_slight_y, anchor_center, tolerance_x=25.0, tolerance_y=25.0)
        self.assertEqual(dir_slight, "to the left of")

        # Both differences within tolerance -> right next to
        target_very_close = (305.0, 305.0)
        dir_close = determine_relative_direction(target_very_close, anchor_center, tolerance_x=25.0, tolerance_y=25.0)
        self.assertEqual(dir_close, "right next to")


class TestAnchorObjectSelection(unittest.TestCase):
    """Test selection of reference/anchor objects."""

    def setUp(self):
        self.target_bottle = DetectionItem(
            class_id=39,
            class_name="bottle",
            confidence=0.95,
            bbox=BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=250.0, width=50.0, height=150.0),
        )

    def test_excludes_same_class_as_target(self):
        another_bottle = DetectionItem(
            class_id=39,
            class_name="bottle",
            confidence=0.90,
            bbox=BoundingBox(x1=200.0, y1=100.0, x2=250.0, y2=250.0, width=50.0, height=150.0),
        )
        chair = DetectionItem(
            class_id=56,
            class_name="chair",
            confidence=0.88,
            bbox=BoundingBox(x1=300.0, y1=100.0, x2=450.0, y2=300.0, width=150.0, height=200.0),
        )

        anchor = select_anchor_object(
            target_item=self.target_bottle,
            all_detections=[self.target_bottle, another_bottle, chair],
            target_class="bottle",
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.class_name, "chair")

    def test_prefers_closer_object(self):
        near_chair = DetectionItem(
            class_id=56,
            class_name="chair",
            confidence=0.85,
            bbox=BoundingBox(x1=200.0, y1=100.0, x2=320.0, y2=250.0, width=120.0, height=150.0),
        )
        far_table = DetectionItem(
            class_id=60,
            class_name="dining table",
            confidence=0.86,
            bbox=BoundingBox(x1=800.0, y1=600.0, x2=1100.0, y2=900.0, width=300.0, height=300.0),
        )

        anchor = select_anchor_object(
            target_item=self.target_bottle,
            all_detections=[self.target_bottle, near_chair, far_table],
            target_class="bottle",
            frame_width=1280,
            frame_height=720,
        )
        self.assertIsNotNone(anchor)
        self.assertEqual(anchor.class_name, "chair")

    def test_filters_out_low_confidence_objects(self):
        low_conf_chair = DetectionItem(
            class_id=56,
            class_name="chair",
            confidence=0.20,  # Below default min_confidence=0.30
            bbox=BoundingBox(x1=200.0, y1=100.0, x2=320.0, y2=250.0, width=120.0, height=150.0),
        )

        anchor = select_anchor_object(
            target_item=self.target_bottle,
            all_detections=[self.target_bottle, low_conf_chair],
            target_class="bottle",
            config=AnchorSelectionConfig(min_confidence=0.30),
        )
        self.assertIsNone(anchor)


class TestRelativeSpatialPipeline(unittest.TestCase):
    """Test full analyze_relative_spatial_relations pipeline scenarios."""

    def test_target_and_anchor_scenario(self):
        bottle = DetectionItem(
            class_id=39,
            class_name="bottle",
            confidence=0.92,
            bbox=BoundingBox(x1=100.0, y1=200.0, x2=160.0, y2=350.0, width=60.0, height=150.0),
        )
        chair = DetectionItem(
            class_id=56,
            class_name="chair",
            confidence=0.89,
            bbox=BoundingBox(x1=300.0, y1=200.0, x2=450.0, y2=400.0, width=150.0, height=200.0),
        )

        res = analyze_relative_spatial_relations("bottle", [bottle, chair])
        self.assertTrue(res.target_found)
        self.assertEqual(res.target_count, 1)
        self.assertFalse(res.multiple_targets)
        self.assertIsNotNone(res.anchor)
        self.assertEqual(res.anchor.class_name, "chair")
        self.assertEqual(res.direction, "to the left of")
        self.assertEqual(res.description, "The bottle is to the left of the chair.")

    def test_no_anchor_available_fallback(self):
        bottle = DetectionItem(
            class_id=39,
            class_name="bottle",
            confidence=0.92,
            bbox=BoundingBox(x1=100.0, y1=200.0, x2=160.0, y2=350.0, width=60.0, height=150.0),
        )

        res = analyze_relative_spatial_relations("bottle", [bottle])
        self.assertTrue(res.target_found)
        self.assertIsNone(res.anchor)
        self.assertEqual(res.description, "I can see a bottle, but I could not find a suitable reference object.")

    def test_target_not_detected(self):
        chair = DetectionItem(
            class_id=56,
            class_name="chair",
            confidence=0.89,
            bbox=BoundingBox(x1=300.0, y1=200.0, x2=450.0, y2=400.0, width=150.0, height=200.0),
        )

        res = analyze_relative_spatial_relations("bottle", [chair])
        self.assertFalse(res.target_found)
        self.assertEqual(res.target_count, 0)
        self.assertEqual(res.description, "I could not detect any bottle in the image.")

    def test_multiple_targets_handling(self):
        bottle1 = DetectionItem(
            class_id=39,
            class_name="bottle",
            confidence=0.95,
            bbox=BoundingBox(x1=100.0, y1=200.0, x2=160.0, y2=350.0, width=60.0, height=150.0),
        )
        bottle2 = DetectionItem(
            class_id=39,
            class_name="bottle",
            confidence=0.80,
            bbox=BoundingBox(x1=500.0, y1=200.0, x2=560.0, y2=350.0, width=60.0, height=150.0),
        )
        chair = DetectionItem(
            class_id=56,
            class_name="chair",
            confidence=0.89,
            bbox=BoundingBox(x1=300.0, y1=200.0, x2=450.0, y2=400.0, width=150.0, height=200.0),
        )

        res = analyze_relative_spatial_relations("bottle", [bottle1, bottle2, chair])
        self.assertTrue(res.target_found)
        self.assertEqual(res.target_count, 2)
        self.assertTrue(res.multiple_targets)
        self.assertIsNotNone(res.anchor)
        self.assertEqual(res.anchor.class_name, "chair")
        self.assertIn("I found 2 bottles", res.description)
        self.assertIn("to the left of the chair", res.description)


if __name__ == "__main__":
    unittest.main()
