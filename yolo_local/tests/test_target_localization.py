import unittest
import numpy as np
from pathlib import Path

from backend.schemas.detection import BoundingBox, DetectionItem
from backend.services.detector import get_default_detector
from backend.services.image_service import ImageService
from backend.services.nlp_extractor import get_nlp_extractor
from backend.services.visualizer import Visualizer
from backend.services.webcam_service import WebcamService


class TestTargetLocalization(unittest.TestCase):
    """
    Test suite verifying that when NLP extracts a target object from speech/text,
    ONLY the target object is located, tracked, and highlighted.
    """

    def setUp(self):
        self.nlp = get_nlp_extractor()
        self.detector = get_default_detector()
        self.webcam_service = WebcamService(detector=self.detector)

    def test_nlp_extracts_target_classes_and_ids(self):
        """Verify NLP accurately identifies target classes and corresponding COCO IDs."""
        res = self.nlp.extract("Where is my cell phone and water bottle?")
        self.assertTrue(res.success)
        self.assertIn("cell phone", res.target_classes)
        self.assertIn("bottle", res.target_classes)
        self.assertIn(67, res.target_class_ids)
        self.assertIn(39, res.target_class_ids)

    def test_webcam_service_set_and_clear_target_filter(self):
        """Verify WebcamService updates target lock state correctly."""
        # Test setting target filter from natural query
        status = self.webcam_service.set_target_filter("Locate the backpack and laptop")
        self.assertTrue(status["target_locked"])
        self.assertIn("backpack", status["active_targets"])
        self.assertIn("laptop", status["active_targets"])
        self.assertEqual(set(status["active_target_ids"]), {24, 63})

        # Test querying status
        current = self.webcam_service.get_target_status()
        self.assertTrue(current["target_locked"])
        self.assertEqual(current["active_targets"], ["backpack", "laptop"])

        # Test clearing target filter
        cleared = self.webcam_service.clear_target_filter()
        self.assertFalse(cleared["target_locked"])
        self.assertEqual(cleared["active_targets"], [])
        self.assertEqual(cleared["active_target_ids"], [])

    def test_query_target_automatically_activates_target_lock(self):
        """Verify WebcamService.query_target automatically locks on extracted targets."""
        self.webcam_service.clear_target_filter()
        res = self.webcam_service.query_target("Where is my coffee mug?")
        self.assertTrue(res["target_locked"])
        self.assertIn("cup", res["target_classes"])
        self.assertTrue(self.webcam_service.target_filter_enabled)
        self.assertIn(41, self.webcam_service.active_target_ids)

    def test_detections_to_telemetry_conversion(self):
        """Verify Visualizer.detections_to_telemetry generates correct spatial geometry."""
        dummy_detections = [
            DetectionItem(
                class_id=67,
                class_name="cell phone",
                confidence=0.95,
                bbox=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0, width=100.0, height=200.0),
            )
        ]
        telemetry = Visualizer.detections_to_telemetry(dummy_detections, (720, 1280, 3))
        self.assertEqual(len(telemetry), 1)
        item = telemetry[0]
        self.assertEqual(item["class_name"], "cell phone")
        self.assertEqual(item["class_id"], 67)
        self.assertEqual(item["confidence"], 95.0)
        self.assertIn("azimuth_deg", item)
        self.assertIn("sector", item)
        self.assertIn("distance_ft", item)

    def test_visualizer_draw_target_overlay_runs(self):
        """Verify Visualizer.draw_target_overlay produces an annotated frame with HUD badges."""
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        target_items = [
            {
                "class_name": "bottle",
                "class_id": 39,
                "confidence": 94.5,
                "azimuth_deg": -14.2,
                "sector": "Left (-14.2°)",
                "distance_ft": 2.4,
                "distance_m": 0.7,
                "bbox": {"x1": 80, "y1": 100, "x2": 160, "y2": 320},
            }
        ]
        annotated = Visualizer.draw_target_overlay(
            frame=canvas,
            target_items=target_items,
            active_targets=["bottle"],
            fps=42.0,
        )
        self.assertEqual(annotated.shape, canvas.shape)
        # Verify frame has been modified (not pure black)
        self.assertGreater(np.count_nonzero(annotated), 0)

    def test_detector_target_filtering_on_sample_image(self):
        """Verify detector with target_ids returns ONLY target objects, suppressing all others."""
        bus_image_path = Path(__file__).resolve().parent.parent / "bus.jpg"
        if not bus_image_path.exists():
            self.skipTest("bus.jpg not found")

        import cv2
        img = cv2.imread(str(bus_image_path))

        # Detect all objects without filter
        all_detections, _, _ = self.detector.detect(img, conf_threshold=0.4, annotate=False)
        all_classes = {d.class_name for d in all_detections}
        # bus.jpg has both 'bus' and 'person'
        self.assertIn("bus", all_classes)
        self.assertIn("person", all_classes)

        # Detect ONLY target: bus (class_id=5)
        bus_detections, _, _ = self.detector.detect(
            img, conf_threshold=0.4, annotate=False, classes=[5], target_ids={5}
        )
        for d in bus_detections:
            self.assertEqual(d.class_name, "bus")
            self.assertEqual(d.class_id, 5)
        self.assertGreater(len(bus_detections), 0)

        # Detect ONLY target: person (class_id=0)
        person_detections, _, _ = self.detector.detect(
            img, conf_threshold=0.4, annotate=False, classes=[0], target_ids={0}
        )
        for d in person_detections:
            self.assertEqual(d.class_name, "person")
            self.assertEqual(d.class_id, 0)
        self.assertGreater(len(person_detections), 0)

    def test_image_service_target_localization(self):
        """Verify ImageService locates only target objects when target_ids is provided."""
        bus_image_path = Path(__file__).resolve().parent.parent / "bus.jpg"
        if not bus_image_path.exists():
            self.skipTest("bus.jpg not found")

        img_service = ImageService(detector=self.detector)
        res, annotated, _ = img_service.process_image_file(
            image_path=bus_image_path,
            conf_threshold=0.4,
            save_output=False,
            target_ids={5},
            target_names=["bus"],
        )
        self.assertTrue(res.success)
        self.assertGreater(res.num_detections, 0)
        for d in res.detections:
            self.assertEqual(d.class_name, "bus")


if __name__ == "__main__":
    unittest.main()
