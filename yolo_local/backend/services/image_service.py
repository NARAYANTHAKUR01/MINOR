from pathlib import Path
import uuid
import cv2
import numpy as np

from backend.config import settings
from backend.schemas.detection import DetectionResultResponse
from backend.services.detector import ObjectDetector, get_default_detector
from backend.services.visualizer import Visualizer


class ImageService:
    """Handles image loading, inference, annotation, and persistence."""

    def __init__(self, detector: ObjectDetector | None = None):
        self._detector = detector

    @property
    def detector(self) -> ObjectDetector:
        if self._detector is None:
            self._detector = get_default_detector()
        return self._detector

    def process_image_file(
        self,
        image_path: Path,
        conf_threshold: float = 0.4,
        output_dir: Path | None = None,
        save_output: bool = True,
        target_ids: set[int] | list[int] | None = None,
        target_names: list[str] | None = None,
    ) -> tuple[DetectionResultResponse, np.ndarray, Path | None]:
        """Process an image from disk, optionally locating ONLY target objects."""
        target_dir = output_dir or settings.outputs_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not decode image at: {image_path}")

        is_target_mode = target_ids is not None
        classes_arg = list(target_ids) if (is_target_mode and target_ids) else None

        detections, elapsed_ms, annotated = self.detector.detect(
            img,
            conf_threshold=conf_threshold,
            annotate=(not is_target_mode),
            classes=classes_arg,
            target_ids=target_ids,
        )

        if is_target_mode:
            telemetry_items = Visualizer.detections_to_telemetry(detections, img.shape)
            annotated = Visualizer.draw_target_overlay(
                frame=img,
                target_items=telemetry_items,
                active_targets=target_names or [],
                fps=0.0,
            )

        saved_path = None
        if save_output and annotated is not None:
            out_filename = f"annotated_{image_path.name}"
            saved_path = target_dir / out_filename
            cv2.imwrite(str(saved_path), annotated)

        response = DetectionResultResponse(
            success=True,
            inference_time_ms=round(elapsed_ms, 2),
            num_detections=len(detections),
            device_used=self.detector.device.upper(),
            detections=detections,
            annotated_image_url=(
                f"/outputs/{saved_path.name}" if saved_path else None
            ),
        )

        return response, annotated, saved_path

    def process_image_bytes(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        conf_threshold: float = 0.4,
        target_ids: set[int] | list[int] | None = None,
        target_names: list[str] | None = None,
    ) -> tuple[DetectionResultResponse, np.ndarray, Path]:
        """Process raw image bytes from an HTTP file upload, optionally locating ONLY targets."""
        target_dir = settings.outputs_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file: Failed to decode bytes.")

        is_target_mode = target_ids is not None
        classes_arg = list(target_ids) if (is_target_mode and target_ids) else None

        detections, elapsed_ms, annotated = self.detector.detect(
            img,
            conf_threshold=conf_threshold,
            annotate=(not is_target_mode),
            classes=classes_arg,
            target_ids=target_ids,
        )

        if is_target_mode:
            telemetry_items = Visualizer.detections_to_telemetry(detections, img.shape)
            annotated = Visualizer.draw_target_overlay(
                frame=img,
                target_items=telemetry_items,
                active_targets=target_names or [],
                fps=0.0,
            )

        # Unique filename to avoid collisions
        unique_name = f"annotated_{uuid.uuid4().hex[:8]}_{filename}"
        saved_path = target_dir / unique_name
        cv2.imwrite(str(saved_path), annotated)

        response = DetectionResultResponse(
            success=True,
            inference_time_ms=round(elapsed_ms, 2),
            num_detections=len(detections),
            device_used=self.detector.device.upper(),
            detections=detections,
            annotated_image_url=f"/outputs/{saved_path.name}",
        )

        return response, annotated, saved_path
