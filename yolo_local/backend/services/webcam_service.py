import sys
import threading
import time
from typing import Generator
import cv2
import numpy as np

from backend.config import settings
from backend.services.detector import ObjectDetector, get_default_detector
from backend.services.nlp_extractor import get_nlp_extractor
from backend.services.visualizer import Visualizer


class ThreadedCamera:
    """
    High-performance non-blocking camera capture for macOS.
    Decouples the 30Hz physical shutter from the YOLO GPU inference loop,
    allowing the pipeline to achieve 35-50+ FPS on Apple Silicon MPS.
    """

    def __init__(self, cam_id: int = 0, width: int = 1280, height: int = 720):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

        # Allow sensor warm-up
        for _ in range(6):
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.ret = True
                    self.frame = frame
                    break
            time.sleep(0.1)

        # Fallback to camera index 1 if index 0 fails
        if (not self.ret or self.frame is None) and cam_id == 0:
            print("[INFO] Camera 0 not responding, attempting camera index 1...")
            self.cap.release()
            self.cap = cv2.VideoCapture(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.2)
            for _ in range(6):
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.ret = True
                        self.frame = frame
                        break
                time.sleep(0.1)

        if not self.ret or self.frame is None:
            self._raise_permission_error(cam_id)

        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _raise_permission_error(self, cam_id: int):
        msg = (
            f"\n[ERROR] Unable to access camera at index {cam_id}.\n"
            + "=" * 65
            + "\nmacOS CAMERA PERMISSION NOTICE:\n"
            + "macOS requires explicit camera permissions for applications.\n"
            + "If camera access fails, please check:\n"
            + "  1. System Settings -> Privacy & Security -> Camera.\n"
            + "  2. Ensure your Terminal (Terminal, iTerm2, VS Code, Cursor) is toggled ON.\n"
            + "  3. Ensure no other application (FaceTime, Zoom, Photo Booth) is locking it.\n"
            + "  4. For external or iPhone Continuity cameras, try: python main.py --source webcam --cam-id 1\n"
            + "=" * 65
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(f"Could not open camera at index {cam_id}")

    def _capture_worker(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
            time.sleep(0.002)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self) -> bool:
        return self.running and self.cap.isOpened()

    def release(self):
        self.running = False
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.4)
        self.cap.release()


class WebcamService:
    """Manages real-time webcam capture, FPS estimation, GUI preview, and MJPEG web streaming."""

    def __init__(self, detector: ObjectDetector | None = None):
        self._detector = detector
        self.latest_telemetry: dict = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "num_detections": 0,
            "detections": [],
            "timestamp": time.time(),
            "target_locked": False,
            "active_targets": [],
        }
        self.active_target_name: str | None = None
        self.active_targets: list[str] = []
        self.active_target_ids: set[int] = set()
        self.target_filter_enabled: bool = False
        self._active_cam: ThreadedCamera | None = None
        self._latest_raw_frame: np.ndarray | None = None
        self._frame_lock = threading.Lock()

    @property
    def detector(self) -> ObjectDetector:
        if self._detector is None:
            self._detector = get_default_detector()
        return self._detector

    def get_latest_telemetry(self) -> dict:
        return self.latest_telemetry

    def set_target_filter(self, targets: list[str] | str) -> dict:
        """
        Activates exclusive target filtering so ONLY target objects are detected and displayed.
        Accepts a single target string, sentence, or list of targets.
        """
        nlp = get_nlp_extractor()
        if isinstance(targets, str):
            res = nlp.extract(targets)
            if res.target_classes:
                self.active_targets = res.target_classes
                self.active_target_ids = set(res.target_class_ids)
                self.target_filter_enabled = True
                self.active_target_name = res.target_classes[0]
            else:
                cls_name = targets.strip().lower()
                cls_id = nlp.get_class_id(cls_name)
                if cls_id is not None:
                    self.active_targets = [nlp.id_to_class(cls_id)]
                    self.active_target_ids = {cls_id}
                    self.target_filter_enabled = True
                    self.active_target_name = self.active_targets[0]
                else:
                    self.active_targets = [cls_name] if cls_name else []
                    self.active_target_ids = set()
                    self.target_filter_enabled = bool(cls_name)
                    self.active_target_name = cls_name or None
        else:
            active_names = []
            active_ids = set()
            for t in targets:
                cls_id = nlp.get_class_id(t)
                if cls_id is not None:
                    active_names.append(nlp.id_to_class(cls_id))
                    active_ids.add(cls_id)
                elif t.strip():
                    active_names.append(t.lower().strip())
            self.active_targets = active_names
            self.active_target_ids = active_ids
            self.target_filter_enabled = bool(active_names)
            self.active_target_name = active_names[0] if active_names else None

        return {
            "target_locked": self.target_filter_enabled,
            "active_targets": self.active_targets,
            "active_target_ids": list(self.active_target_ids),
        }

    def clear_target_filter(self) -> dict:
        """Clears active target lock to restore full-scene detection of all classes."""
        self.target_filter_enabled = False
        self.active_targets = []
        self.active_target_ids = set()
        self.active_target_name = None
        return {
            "target_locked": False,
            "active_targets": [],
            "active_target_ids": [],
        }

    def get_target_status(self) -> dict:
        """Returns the current target lock state."""
        return {
            "target_locked": self.target_filter_enabled,
            "active_targets": self.active_targets,
            "active_target_ids": list(self.active_target_ids),
            "active_target_name": self.active_target_name,
        }

    def query_target(self, query_text: str) -> dict:
        nlp = get_nlp_extractor()
        nlp_res = nlp.extract(query_text)
        target_classes = nlp_res.target_classes

        # Automatically engage exclusive target lock for the requested target
        if target_classes:
            self.set_target_filter(target_classes)
        elif query_text.strip():
            self.set_target_filter(query_text.strip())

        detections = self.latest_telemetry.get("detections", [])

        if not target_classes:
            # Fallback if query didn't contain known COCO objects
            q = query_text.lower().strip()
            matched = [
                d for d in detections
                if q in d["class_name"].lower() or d["class_name"].lower() in q
            ]
        else:
            matched = [
                d for d in detections
                if d["class_name"].lower() in target_classes
            ]

        if matched:
            best = max(matched, key=lambda x: x["confidence"])
            self.active_target_name = best["class_name"]
            speech = f"{best['class_name'].capitalize()} spotted {best['distance_ft']} feet away in your {best['sector']}."
            return {
                "found": True,
                "target": best["class_name"],
                "target_classes": target_classes,
                "target_locked": self.target_filter_enabled,
                "confidence": best["confidence"],
                "azimuth_deg": best["azimuth_deg"],
                "sector": best["sector"],
                "distance_ft": best["distance_ft"],
                "distance_m": best["distance_m"],
                "bbox": best["bbox"],
                "speech_text": speech,
                "nlp_info": {
                    "intent": nlp_res.intent,
                    "target_classes": nlp_res.target_classes,
                    "target_class_ids": nlp_res.target_class_ids,
                    "excluded_classes": nlp_res.excluded_classes,
                },
            }

        target_display = ", ".join(target_classes) if target_classes else query_text
        return {
            "found": False,
            "target": target_display,
            "target_classes": target_classes,
            "target_locked": self.target_filter_enabled,
            "speech_text": f"Scanning for {target_display}. Not currently in camera view.",
            "nlp_info": {
                "intent": nlp_res.intent,
                "target_classes": nlp_res.target_classes,
                "target_class_ids": nlp_res.target_class_ids,
                "excluded_classes": nlp_res.excluded_classes,
            },
        }

    def run_live_preview(
        self,
        cam_id: int = 0,
        conf_threshold: float = 0.4,
        show_preview: bool = True,
        imgsz: int = 512,
    ) -> None:
        """
        Run real-time webcam detection with non-blocking threaded capture.
        Default imgsz=512 hits 35-45+ FPS on Apple Silicon MPS.
        """
        cam = ThreadedCamera(cam_id)
        window_name = "YOLO11m Live Detection (Press 'q' or Esc to quit)"
        print(f"[INFO] Threaded camera active. Inference resolution: {imgsz}x{imgsz} on Apple Silicon MPS.")
        print("[INFO] Tip: Press 'q' or 'Esc' in the preview window to exit.")

        prev_time = time.time()
        fps_smooth = 0.0
        last_log_time = 0.0

        try:
            while True:
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                instant_fps = (1.0 / dt) if dt > 0 else 0.0
                fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

                if self.target_filter_enabled:
                    classes_arg = list(self.active_target_ids) if self.active_target_ids else None
                    target_ids_arg = self.active_target_ids
                    annotate_arg = False
                else:
                    classes_arg = None
                    target_ids_arg = None
                    annotate_arg = True

                detections, elapsed_ms, annotated = self.detector.detect(
                    frame,
                    conf_threshold=conf_threshold,
                    annotate=annotate_arg,
                    imgsz=imgsz,
                    classes=classes_arg,
                    target_ids=target_ids_arg,
                )

                telemetry_items = Visualizer.detections_to_telemetry(detections, frame.shape)

                self.latest_telemetry = {
                    "fps": round(fps_smooth, 1),
                    "latency_ms": round(elapsed_ms, 1),
                    "num_detections": len(detections),
                    "detections": telemetry_items,
                    "timestamp": time.time(),
                    "target_locked": self.target_filter_enabled,
                    "active_targets": self.active_targets,
                }

                if self.target_filter_enabled:
                    output_frame = Visualizer.draw_target_overlay(
                        frame=frame,
                        target_items=telemetry_items,
                        active_targets=self.active_targets,
                        fps=fps_smooth,
                    )
                else:
                    output_frame = annotated if annotated is not None else frame
                    class_counts: dict[str, int] = {}
                    for d in detections:
                        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
                    summary_items = [
                        f"{name} x{cnt}" if cnt > 1 else name
                        for name, cnt in sorted(class_counts.items())
                    ]
                    summary_str = ", ".join(summary_items[:4])
                    if len(summary_items) > 4:
                        summary_str += f" +{len(summary_items) - 4} more"

                    output_frame = Visualizer.draw_fps(
                        output_frame,
                        fps_smooth,
                        count=len(detections),
                        summary=summary_str,
                    )

                # Throttled console log (every 2.5s)
                if curr_time - last_log_time > 2.5:
                    if detections:
                        print(f"[LIVE] FPS: {fps_smooth:4.1f} | Objects detected ({len(detections)}): {summary_str}")
                    else:
                        print(f"[LIVE] FPS: {fps_smooth:4.1f} | Scanning environment (no objects above conf {conf_threshold})")
                    last_log_time = curr_time

                if show_preview:
                    try:
                        cv2.imshow(window_name, output_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q"), 27):
                            print("\n[INFO] Exiting webcam stream ('q'/Esc pressed).")
                            break
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            print("\n[INFO] Preview window closed.")
                            break
                    except Exception as e:
                        print(f"[ERROR] GUI preview error: {e}", file=sys.stderr)
                        break
        finally:
            cam.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera released and windows closed.")

    def generate_mjpeg_stream(
        self,
        cam_id: int = 0,
        conf_threshold: float = 0.4,
        imgsz: int = 512,
    ) -> Generator[bytes, None, None]:
        """
        Yields multipart MJPEG frames for HTTP streaming directly to a web browser or frontend <img> tag.
        """
        cam = ThreadedCamera(cam_id)
        with self._frame_lock:
            self._active_cam = cam
        prev_time = time.time()
        fps_smooth = 0.0

        try:
            while True:
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                with self._frame_lock:
                    self._latest_raw_frame = frame

                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                instant_fps = (1.0 / dt) if dt > 0 else 0.0
                fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps if fps_smooth > 0 else instant_fps

                if self.target_filter_enabled:
                    classes_arg = list(self.active_target_ids) if self.active_target_ids else None
                    target_ids_arg = self.active_target_ids
                    annotate_arg = False
                else:
                    classes_arg = None
                    target_ids_arg = None
                    annotate_arg = True

                detections, elapsed_ms, annotated = self.detector.detect(
                    frame,
                    conf_threshold=conf_threshold,
                    annotate=annotate_arg,
                    imgsz=imgsz,
                    classes=classes_arg,
                    target_ids=target_ids_arg,
                )

                telemetry_items = Visualizer.detections_to_telemetry(detections, frame.shape)

                self.latest_telemetry = {
                    "fps": round(fps_smooth, 1),
                    "latency_ms": round(elapsed_ms, 1),
                    "num_detections": len(detections),
                    "detections": telemetry_items,
                    "timestamp": time.time(),
                    "target_locked": self.target_filter_enabled,
                    "active_targets": self.active_targets,
                }

                if self.target_filter_enabled:
                    output_frame = Visualizer.draw_target_overlay(
                        frame=frame,
                        target_items=telemetry_items,
                        active_targets=self.active_targets,
                        fps=fps_smooth,
                    )
                else:
                    output_frame = annotated if annotated is not None else frame
                    class_counts: dict[str, int] = {}
                    for d in detections:
                        class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
                    summary_items = [
                        f"{name} x{cnt}" if cnt > 1 else name
                        for name, cnt in sorted(class_counts.items())
                    ]
                    summary_str = ", ".join(summary_items[:3])

                    output_frame = Visualizer.draw_fps(
                        output_frame,
                        fps_smooth,
                        count=len(detections),
                        summary=summary_str,
                    )

                # JPEG encode
                success, buffer = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not success:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
        finally:
            with self._frame_lock:
                self._active_cam = None
                self._latest_raw_frame = None
            cam.release()

    def get_latest_frame(self, cam_id: int = 0) -> np.ndarray | None:
        """
        Thread-safe accessor for the latest raw camera frame.
        If live stream is active, reuses the active capture frame.
        If idle, temporarily captures a single frame.
        """
        with self._frame_lock:
            if self._active_cam is not None:
                ret, frame = self._active_cam.read()
                if ret and frame is not None:
                    return frame.copy()
            if self._latest_raw_frame is not None:
                return self._latest_raw_frame.copy()

        # If camera is not currently streaming, open temporarily for one snapshot (throttled)
        now = time.time()
        if hasattr(self, "_last_cap_failed_time") and (now - self._last_cap_failed_time < 0.5):
            return None

        try:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened() and cam_id == 0:
                cap.release()
                cap = cv2.VideoCapture(1)
            if cap.isOpened():
                for _ in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.release()
                        return frame
                    time.sleep(0.02)
                cap.release()
            self._last_cap_failed_time = time.time()
        except Exception as e:
            self._last_cap_failed_time = time.time()
            print(f"[WARN] Failed to capture standalone frame: {e}")
        return None

