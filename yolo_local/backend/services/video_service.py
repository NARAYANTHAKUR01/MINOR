from pathlib import Path
import sys
import time
from typing import Callable
import cv2
import numpy as np

from backend.config import settings
from backend.services.detector import ObjectDetector, get_default_detector
from backend.services.visualizer import Visualizer


class VideoService:
    """Handles video file reading, frame-by-frame inference, and MP4 generation."""

    def __init__(self, detector: ObjectDetector | None = None):
        self._detector = detector

    @property
    def detector(self) -> ObjectDetector:
        if self._detector is None:
            self._detector = get_default_detector()
        return self._detector

    def process_video_file(
        self,
        video_path: Path,
        conf_threshold: float = 0.4,
        output_dir: Path | None = None,
        show_preview: bool = True,
        progress_callback: Callable[[int, int, float], None] | None = None,
        target_ids: set[int] | list[int] | None = None,
        target_names: list[str] | None = None,
    ) -> Path:
        """
        Processes a video file frame-by-frame and exports an annotated MP4.
        If target_ids is specified, ONLY the target objects are located and highlighted.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_dir = output_dir or settings.outputs_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"annotated_{video_path.stem}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        is_target_mode = target_ids is not None
        classes_arg = list(target_ids) if (is_target_mode and target_ids) else None

        print(
            f"[INFO] Video input: {video_path.name} ({width}x{height} @ {fps:.1f} FPS, {total_frames} frames)"
        )
        if is_target_mode:
            print(f"[INFO] Exclusive Target Localization active: {target_names or target_ids}")
        if show_preview:
            print("[INFO] Live preview enabled. Press 'q' in the preview window to stop early.")

        frame_idx = 0
        start_total = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                # Detect and annotate
                detections, _, annotated = self.detector.detect(
                    frame,
                    conf_threshold=conf_threshold,
                    annotate=(not is_target_mode),
                    classes=classes_arg,
                    target_ids=target_ids,
                )
                if is_target_mode:
                    telemetry_items = Visualizer.detections_to_telemetry(detections, frame.shape)
                    frame_to_write = Visualizer.draw_target_overlay(
                        frame=frame,
                        target_items=telemetry_items,
                        active_targets=target_names or [],
                        fps=fps,
                    )
                else:
                    frame_to_write = annotated if annotated is not None else frame

                writer.write(frame_to_write)

                if show_preview:
                    try:
                        cv2.imshow(
                            "YOLO11m Video Detection (Press 'q' to quit)", frame_to_write
                        )
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("\n[INFO] User requested early quit ('q' pressed).")
                            break
                    except Exception as exc:
                        print(f"\n[WARNING] Preview window disabled: {exc}")
                        show_preview = False

                if progress_callback:
                    progress_callback(frame_idx, total_frames, (frame_idx / total_frames) * 100.0)
                elif total_frames > 0 and frame_idx % 15 == 0:
                    pct = (frame_idx / total_frames) * 100.0
                    sys.stdout.write(f"\r[INFO] Processing: frame {frame_idx}/{total_frames} ({pct:5.1f}%)")
                    sys.stdout.flush()
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

        elapsed = time.time() - start_total
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"\n[SUCCESS] Video complete! {frame_idx} frames processed in {elapsed:.1f}s ({avg_fps:.1f} FPS).")
        print(f"[SUCCESS] Saved to: {out_path.resolve()}")
        return out_path
