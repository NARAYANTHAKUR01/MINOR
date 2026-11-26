import time
from pathlib import Path
import cv2
import numpy as np
import torch

from backend.config import settings
from backend.core.model_loader import get_yolo_model
from backend.schemas.detection import BoundingBox, DetectionItem
from backend.services.visualizer import Visualizer


class ObjectDetector:
    """Wrapper around YOLO11m providing structured detection results with high accuracy."""

    def __init__(self, weights_path: Path | None = None, device_preference: str = "auto"):
        self.weights_path = weights_path or settings.default_model_path
        self.model, self.device = get_yolo_model(self.weights_path, device_preference)
        self.class_names: dict[int, str] = self.model.names

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        annotate: bool = False,
        imgsz: int = 640,
        classes: list[int] | None = None,
        target_ids: set[int] | list[int] | None = None,
        iou_threshold: float = 0.45,
        augment: bool = False,
        enhance_lighting: bool = True,
    ) -> tuple[list[DetectionItem], float, np.ndarray | None]:
        """
        Run inference on an image/frame with optimized resolution and NMS.
        If classes or target_ids are specified, filters ONLY for target objects.

        Returns:
            - List of DetectionItem schemas
            - Inference duration in milliseconds
            - Annotated image if annotate=True, otherwise None
        """
        # Adaptive lighting enhancement for dim or shadowed camera frames
        processed_frame = frame
        if enhance_lighting and frame is not None and frame.size > 0:
            try:
                if frame.ndim == 3 and frame.shape[2] == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if float(np.mean(gray)) < 65:
                        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                        l_chan, a_chan, b_chan = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l_enh = clahe.apply(l_chan)
                        processed_frame = cv2.cvtColor(cv2.merge((l_enh, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
            except Exception:
                processed_frame = frame

        start = time.perf_counter()
        predict_kwargs = {
            "source": processed_frame,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "imgsz": imgsz,
            "device": self.device,
            "verbose": False,
            "max_det": 100,
        }
        if augment:
            predict_kwargs["augment"] = True
        if classes is not None:
            predict_kwargs["classes"] = classes

        with torch.inference_mode():
            results = self.model.predict(**predict_kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        res = results[0]
        detections: list[DetectionItem] = []

        active_target_filter = set(target_ids) if target_ids is not None else None

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                if active_target_filter is not None and cls_id not in active_target_filter:
                    continue

                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [round(float(c), 2) for c in coords]
                conf = round(float(box.conf[0]), 4)
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

                detections.append(
                    DetectionItem(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=BoundingBox(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            width=round(x2 - x1, 2),
                            height=round(y2 - y1, 2),
                        ),
                    )
                )

        annotated_frame = None
        if annotate:
            annotated_frame = Visualizer.draw_detections(
                frame, res, self.class_names, target_ids=active_target_filter
            )

        return detections, elapsed_ms, annotated_frame


_DEFAULT_DETECTOR: ObjectDetector | None = None


def get_default_detector() -> ObjectDetector:
    """Return or create a shared singleton ObjectDetector instance."""
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = ObjectDetector()
    return _DEFAULT_DETECTOR

