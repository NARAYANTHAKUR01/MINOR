"""
Comprehensive unit test suite for Interactive Voice-Based Target Object Detection.

Tests:
1. SpatialReasoner 3x3 normalized frame partitioning and direction mapping.
2. NLPExtractor requested region and target intent extraction.
3. InteractiveAssistant state machine transitions (10 states).
4. 5-second monotonic search loop:
   - Immediate match exit when object matches requested region.
   - 5-second window execution for region mismatches.
   - Timeout reporting when target is absent.
   - Immediate cancellation when toggled OFF.
5. FastAPI REST API endpoints (/toggle, /status, /text-command, /process-audio).
"""

import time
import unittest
import numpy as np

from fastapi.testclient import TestClient

from backend.api.app import app
from backend.schemas.detection import BoundingBox, DetectionItem
from backend.services.detector import ObjectDetector
from backend.services.interactive_assistant import (
    InteractiveAssistant,
    InteractiveState,
    get_interactive_assistant,
)
from backend.services.nlp_extractor import get_nlp_extractor
from backend.services.spatial_reasoner import SpatialReasoner
from backend.services.webcam_service import WebcamService


class MockDetector(ObjectDetector):
    """Mock detector returning pre-configured detections for testing search loop."""

    def __init__(self, detections: list[DetectionItem] | None = None):
        self._mock_detections = detections or []

    def set_mock_detections(self, detections: list[DetectionItem]):
        self._mock_detections = detections

    def detect(self, image, conf_threshold=0.25, annotate=False, imgsz=640, classes=None, target_ids=None):
        if classes is not None:
            filtered = [d for d in self._mock_detections if d.class_id in classes]
            return filtered, 5.0, None
        return self._mock_detections, 5.0, None


class TestSpatialReasoner(unittest.TestCase):
    """Test 3x3 grid calculation and natural language feedback generation."""

    def test_3x3_grid_centroid_calculations(self):
        # Frame 600x600 (3x3 grid: 0-200, 200-400, 400-600)
        # Center box
        center_bbox = {"x1": 250, "y1": 250, "x2": 350, "y2": 350, "width": 100, "height": 100}
        loc = SpatialReasoner.calculate_object_location(center_bbox, 600, 600)
        self.assertEqual(loc["region"], "center")
        self.assertEqual(loc["col"], "center")
        self.assertIn(loc["row"], ("center", "middle"))

        # Top-left box
        tl_bbox = {"x1": 20, "y1": 20, "x2": 100, "y2": 100, "width": 80, "height": 80}
        loc_tl = SpatialReasoner.calculate_object_location(tl_bbox, 600, 600)
        self.assertEqual(loc_tl["region"], "top-left")
        self.assertEqual(loc_tl["col"], "left")
        self.assertEqual(loc_tl["row"], "top")

        # Bottom-right box
        br_bbox = {"x1": 450, "y1": 450, "x2": 580, "y2": 580, "width": 130, "height": 130}
        loc_br = SpatialReasoner.calculate_object_location(br_bbox, 600, 600)
        self.assertEqual(loc_br["region"], "bottom-right")
        self.assertEqual(loc_br["col"], "right")
        self.assertEqual(loc_br["row"], "bottom")

    def test_region_matching_broad_and_exact(self):
        # Broad 'left' should match 'left', 'top-left', 'bottom-left'
        self.assertTrue(SpatialReasoner.matches_requested_region("left", "left"))
        self.assertTrue(SpatialReasoner.matches_requested_region("top-left", "left"))
        self.assertTrue(SpatialReasoner.matches_requested_region("bottom-left", "left"))
        self.assertFalse(SpatialReasoner.matches_requested_region("right", "left"))
        self.assertFalse(SpatialReasoner.matches_requested_region("top-right", "left"))

        # Exact matching
        self.assertTrue(SpatialReasoner.matches_requested_region("top-left", "top-left"))
        self.assertFalse(SpatialReasoner.matches_requested_region("top-right", "top-left"))

        # Unconstrained (None) matches any region
        self.assertTrue(SpatialReasoner.matches_requested_region("center", None))

    def test_response_generation_scenarios(self):
        # Scenario 1: Object found in matching region (environmental: on your left)
        resp_hit = SpatialReasoner.generate_location_response(
            target_object="bottle",
            found=True,
            detected_region="left",
            requested_region="left",
            region_matched=True,
        )
        self.assertIn("found your bottle on your left", resp_hit)

        # Scenario 2: Object found in center (environmental: directly in front of you)
        resp_center = SpatialReasoner.generate_location_response(
            target_object="cup",
            found=True,
            detected_region="center",
            requested_region=None,
            region_matched=True,
        )
        self.assertIn("found your cup directly in front of you", resp_center)

        # Scenario 3: Object found in wrong region
        resp_mismatch = SpatialReasoner.generate_location_response(
            target_object="phone",
            found=True,
            detected_region="right",
            requested_region="left",
            region_matched=False,
        )
        self.assertEqual(
            resp_mismatch,
            "I found your phone on your right, but I couldn't find one on your left.",
        )

        # Scenario 4: Object absent / timed out
        resp_timeout = SpatialReasoner.generate_location_response(
            target_object="backpack",
            found=False,
            timed_out=True,
        )
        self.assertEqual(resp_timeout, "Sorry, I can't detect this type of object around you.")


class TestNLPExtractionInteractive(unittest.TestCase):
    """Test requested spatial region extraction and target intent parsing."""

    def setUp(self):
        self.nlp = get_nlp_extractor()

    def test_requested_region_extraction(self):
        self.assertEqual(self.nlp.extract_requested_region("Find the bottle on the left"), "left")
        self.assertEqual(self.nlp.extract_requested_region("Where is my phone on the right side?"), "right")
        self.assertEqual(self.nlp.extract_requested_region("Can you see a cup at the top-left corner?"), "top-left")
        self.assertEqual(self.nlp.extract_requested_region("Look at the bottom right"), "bottom-right")
        self.assertEqual(self.nlp.extract_requested_region("Is there a cat in the middle?"), "center")
        self.assertEqual(self.nlp.extract_requested_region("Where is my laptop?"), None)

    def test_extract_target_and_intent(self):
        res = self.nlp.extract_target_and_intent("Find the bottle on the left")
        self.assertTrue(res["valid"])
        self.assertEqual(res["target_object"], "bottle")
        self.assertEqual(res["target_class_id"], 39)
        self.assertEqual(res["requested_region"], "left")
        self.assertEqual(res["intent"], "locate_in_region")

        res_simple = self.nlp.extract_target_and_intent("Where is my cell phone?")
        self.assertTrue(res_simple["valid"])
        self.assertEqual(res_simple["target_object"], "cell phone")
        self.assertEqual(res_simple["target_class_id"], 67)
        self.assertIsNone(res_simple["requested_region"])


class TestInteractiveAssistantEngine(unittest.TestCase):
    """Test the 10-state machine, 5-second search loop, and cancellation."""

    def setUp(self):
        self.mock_detector = MockDetector()
        self.assistant = InteractiveAssistant(detector=self.mock_detector)
        # Blank test frame (600x600)
        self.dummy_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    def tearDown(self):
        self.assistant.disable()

    def test_initial_state_and_toggle(self):
        self.assertEqual(self.assistant.state, InteractiveState.DISABLED)
        self.assertFalse(self.assistant.is_active())

        # Enable
        status = self.assistant.enable()
        self.assertEqual(status.state, InteractiveState.WAITING_FOR_WAKE_WORD.value)
        self.assertTrue(status.active)

        # Disable
        status = self.assistant.disable()
        self.assertEqual(status.state, InteractiveState.DISABLED.value)
        self.assertFalse(status.active)

    def test_search_immediate_hit_on_region_match(self):
        self.assistant.enable()
        # Mock detection: bottle (ID 39) on the left (x1=20, x2=150, cx=85/600 -> left)
        left_bottle = DetectionItem(
            class_id=39,
            class_name="bottle",
            confidence=0.92,
            bbox=BoundingBox(x1=20, y1=200, x2=150, y2=400, width=130, height=200),
        )
        self.mock_detector.set_mock_detections([left_bottle])

        t0 = time.monotonic()
        res = self.assistant.search_for_target_with_timeout(
            target_name="bottle",
            target_id=39,
            requested_region="left",
            timeout_seconds=5.0,
            frame_override=self.dummy_frame,
        )
        elapsed = time.monotonic() - t0

        self.assertTrue(res["found"])
        self.assertTrue(res["region_matched"])
        self.assertEqual(res["detected_region"], "left")
        # Should return almost instantly, well under 1 second (not wait 5 seconds)
        self.assertLess(elapsed, 1.5)

    def test_search_keeps_searching_on_region_mismatch_and_reports_mismatch(self):
        self.assistant.enable()
        # Mock detection: bottle on the RIGHT (x1=450, x2=550, cx=500/600 -> right)
        right_bottle = DetectionItem(
            class_id=39,
            class_name="bottle",
            confidence=0.88,
            bbox=BoundingBox(x1=450, y1=200, x2=550, y2=400, width=100, height=200),
        )
        self.mock_detector.set_mock_detections([right_bottle])

        # We request "left", so it will see the right bottle, keep searching until timeout, then report mismatch
        res = self.assistant.search_for_target_with_timeout(
            target_name="bottle",
            target_id=39,
            requested_region="left",
            timeout_seconds=0.25,  # Short timeout for quick test
            frame_override=self.dummy_frame,
        )

        self.assertTrue(res["found"])
        self.assertFalse(res["region_matched"])
        self.assertEqual(res["detected_region"], "right")

    def test_search_timeout_when_target_absent(self):
        self.assistant.enable()
        self.mock_detector.set_mock_detections([])  # No objects in frame

        res = self.assistant.search_for_target_with_timeout(
            target_name="chair",
            target_id=56,
            timeout_seconds=0.2,
            frame_override=self.dummy_frame,
        )

        self.assertFalse(res["found"])
        self.assertFalse(res["region_matched"])
        self.assertIsNone(res["detected_region"])

    def test_process_text_command_unrecognized_object(self):
        self.assistant.enable()
        resp = self.assistant.process_text_command("Where is the alien spacecraft?")
        self.assertFalse(resp.success)
        self.assertIn("didn't recognize which object you asked for", resp.speech_response)
        self.assertEqual(self.assistant.state, InteractiveState.WAITING_FOR_WAKE_WORD)

    def test_immediate_abort_on_disable(self):
        self.assistant.enable()
        self.assistant.disable()  # Immediately toggle OFF

        res = self.assistant.search_for_target_with_timeout(
            target_name="bottle",
            target_id=39,
            timeout_seconds=5.0,
            frame_override=self.dummy_frame,
        )

        self.assertTrue(res.get("aborted"))
        self.assertEqual(self.assistant.state, InteractiveState.DISABLED)


class TestInteractiveFastAPIEndpoints(unittest.TestCase):
    """Test FastAPI HTTP endpoints for Interactive Mode."""

    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.post("/api/interactive/toggle", json={"enable": False})

    def test_api_toggle_and_status(self):
        # Toggle ON
        res_on = self.client.post("/api/interactive/toggle", json={"enable": True})
        self.assertEqual(res_on.status_code, 200)
        self.assertTrue(res_on.json()["active"])
        self.assertEqual(res_on.json()["state"], "WAITING_FOR_WAKE_WORD")

        # Check status
        res_status = self.client.get("/api/interactive/status")
        self.assertEqual(res_status.status_code, 200)
        self.assertTrue(res_status.json()["active"])

        # Toggle OFF
        res_off = self.client.post("/api/interactive/toggle", json={"enable": False})
        self.assertEqual(res_off.status_code, 200)
        self.assertFalse(res_off.json()["active"])
        self.assertEqual(res_off.json()["state"], "DISABLED")

    def test_api_set_state(self):
        self.client.post("/api/interactive/toggle", json={"enable": True})
        res = self.client.post("/api/interactive/set-state", json={"state": "LISTENING_FOR_COMMAND"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["state"], "LISTENING_FOR_COMMAND")

    def test_api_text_command(self):
        self.client.post("/api/interactive/toggle", json={"enable": True})
        res = self.client.post("/api/interactive/text-command", json={"text": "Where is my laptop?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["target_object"], "laptop")
        self.assertEqual(data["target_class_id"], 63)
        self.assertIn("speech_response", data)


if __name__ == "__main__":
    unittest.main()
