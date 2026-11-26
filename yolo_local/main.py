#!/usr/bin/env python3
"""
CLI Runner for YOLO11m Local Object Detection on macOS (Apple Silicon MPS / CPU).

Powered by the modular backend services:
- backend.core: Device selection and weights management
- backend.services: Detection, Visualization, Image, Video, and Webcam services
"""

import argparse
from pathlib import Path
import subprocess
import sys
import cv2

from backend.config import settings
from backend.core.device import get_device
from backend.core.model_loader import ensure_weights
from backend.services.detector import ObjectDetector
from backend.services.image_service import ImageService
from backend.services.nlp_extractor import get_nlp_extractor
from backend.services.video_service import VideoService
from backend.services.webcam_service import WebcamService


def run_image(
    image_service: ImageService,
    path: str,
    conf: float,
    output_dir: Path,
    show_preview: bool,
    target_ids: set[int] | None = None,
    target_names: list[str] | None = None,
    spatial: bool = False,
    tolerance: float = 25.0,
):
    img_path = Path(path)
    if not img_path.is_absolute():
        img_path = Path.cwd() / img_path

    if not img_path.exists():
        print(f"[ERROR] Image file not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    if spatial and target_names:
        target_name = target_names[0]
        print(f"[INFO] Running anchor-based spatial reasoning for target '{target_name}' on '{img_path.name}'...")
        spatial_res, annotated, saved_path = image_service.process_image_spatial_reasoning(
            image_path=img_path,
            target_class=target_name,
            conf_threshold=conf,
            tolerance_x=tolerance,
            tolerance_y=tolerance,
            save_output=True,
            output_dir=output_dir,
        )
        print(f"\n[SPATIAL FEEDBACK] \"{spatial_res.description}\"")
        if spatial_res.anchor:
            print(f"  • Reference Anchor: {spatial_res.anchor.class_name} (conf: {spatial_res.anchor.confidence:.2f})")
            print(f"  • Relative Vector:  {spatial_res.direction}")
            print(f"  • Distance:         {spatial_res.distance_pixels} px")
        print(f"[SUCCESS] Saved annotated spatial image to: {saved_path.resolve() if saved_path else 'None'}")

        # Speak result using TTS
        try:
            subprocess.run(["say", spatial_res.description], check=False)
        except Exception:
            pass

        if show_preview and annotated is not None:
            try:
                window_name = f"Spatial Reasoning - {img_path.name} (Press any key to close)"
                cv2.imshow(window_name, annotated)
                print("[INFO] Showing spatial reasoning window. Press any key to exit.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as exc:
                print(f"[WARNING] Could not open preview window ({exc}).")
        return

    if target_ids is not None:
        print(f"[INFO] Running target localization for {target_names} (IDs: {list(target_ids)}) on '{img_path.name}'...")
    else:
        print(f"[INFO] Running detection on '{img_path.name}' (conf threshold: {conf})...")

    res, annotated, saved_path = image_service.process_image_file(
        image_path=img_path,
        conf_threshold=conf,
        output_dir=output_dir,
        save_output=True,
        target_ids=target_ids,
        target_names=target_names,
    )

    print(f"[INFO] Inference finished in {res.inference_time_ms} ms.")
    print(f"[SUCCESS] Saved annotated image to: {saved_path.resolve() if saved_path else 'None'}")
    if target_ids is not None:
        print(f"[TARGET LOCK] Target objects located: {res.num_detections}")
    else:
        print(f"[INFO] Objects detected: {res.num_detections}")
    for item in res.detections:
        print(f"  - {item.class_name}: {item.confidence:.2f} at [{item.bbox.x1}, {item.bbox.y1}, {item.bbox.x2}, {item.bbox.y2}]")

    if show_preview and annotated is not None:
        try:
            window_name = f"YOLO11m Target Detection - {img_path.name} (Press any key to close)"
            cv2.imshow(window_name, annotated)
            print("[INFO] Showing detection window. Press any key in the window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as exc:
            print(f"[WARNING] Could not open preview window ({exc}). File saved in {saved_path}.")


def run_video(
    video_service: VideoService,
    path: str,
    conf: float,
    output_dir: Path,
    show_preview: bool,
    target_ids: set[int] | None = None,
    target_names: list[str] | None = None,
):
    vid_path = Path(path)
    if not vid_path.is_absolute():
        vid_path = Path.cwd() / vid_path

    if not vid_path.exists():
        print(f"[ERROR] Video file not found: {vid_path}", file=sys.stderr)
        sys.exit(1)

    video_service.process_video_file(
        video_path=vid_path,
        conf_threshold=conf,
        output_dir=output_dir,
        show_preview=show_preview,
        target_ids=target_ids,
        target_names=target_names,
    )


def run_webcam(
    webcam_service: WebcamService,
    cam_id: int,
    conf: float,
    show_preview: bool,
    imgsz: int = 512,
    target_names: list[str] | None = None,
):
    if target_names:
        webcam_service.set_target_filter(target_names)
        print(f"[TARGET LOCK] Active! Locating ONLY: {target_names}")
    try:
        webcam_service.run_live_preview(
            cam_id=cam_id,
            conf_threshold=conf,
            show_preview=show_preview,
            imgsz=imgsz,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Stream stopped by user (Ctrl+C).")


def run_interactive_cli(webcam_service: WebcamService, detector: ObjectDetector, cam_id: int = 0):
    from backend.services.interactive_assistant import get_interactive_assistant
    assistant = get_interactive_assistant()
    assistant.enable()
    print("=" * 65)
    print(" 🎙️  OMNISIGHT INTERACTIVE VOICE ASSISTANT")
    print("=" * 65)
    print(" Say 'Hii' or enter your target question directly.")
    print(" Examples:")
    print("   • 'Hii' or 'Hi' -> Acknowledges and waits for target question")
    print("   • 'Where is my bottle?' -> Searches camera frames for 5.0 seconds")
    print("   • 'bottle' -> Extracts bottle and gives relative spatial description")
    print("   • 'Find phone on the left' -> Searches specifically in requested region")
    print("   • 'q' or 'exit' -> Exit interactive mode")
    print("-" * 65)

    try:
        while True:
            try:
                line = input("\n[Interactive] Enter command or object > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line.lower() in ("q", "quit", "exit"):
                break

            if line.lower() in ("hi", "hii", "hello", "hey", "hi omii", "hi omi", "hey omii", "hey omi", "hello omii"):
                ack = "Yes, I'm listening. Which object should I find?"
                print(f"🗣️  Assistant: \"{ack}\"")
                try:
                    subprocess.run(["say", ack], check=False)
                except Exception:
                    pass
                continue

            resp = assistant.process_text_command(line)
            if resp.target_object:
                print(f"🎯 Target Extracted: {resp.target_object.upper()} (COCO ID: {resp.target_class_id})")
            if resp.requested_region:
                print(f"🧭 Requested Region: {resp.requested_region}")
            print(f"⏱️ Search Duration:  {resp.search_duration_seconds:.2f}s")
            print(f"🗣️ Spoken Feedback:  \"{resp.speech_response}\"")
            try:
                subprocess.run(["say", resp.speech_response], check=False)
            except Exception:
                pass
    finally:
        assistant.disable()
        print("\n[INFO] Interactive mode disabled.")


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO11m object detection fully locally on macOS (Apple Silicon MPS / CPU)."
    )
    parser.add_argument(
        "--source",
        choices=["image", "video", "webcam", "interactive"],
        required=True,
        help="Input source mode: 'image', 'video', 'webcam', or 'interactive'",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to input image or video file (required for 'image' and 'video' modes)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=settings.default_conf_threshold,
        help=f"Confidence threshold for detections (default: {settings.default_conf_threshold})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(settings.default_model_path),
        help=f"Path to local YOLO11m model weights (default: {settings.default_model_path})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.outputs_dir),
        help=f"Directory where annotated images/videos are saved (default: {settings.outputs_dir})",
    )
    parser.add_argument(
        "--cam-id",
        type=int,
        default=settings.default_cam_id,
        help=f"Webcam camera index for 'webcam' mode (default: {settings.default_cam_id})",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Compute device: 'auto' (detects Apple Silicon MPS), 'mps', or 'cpu'",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inference image resolution (default: 512 for webcam [35-45 FPS], 640 for image/video)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="High-speed mode: optimizes resolution (imgsz=480) for 40-50+ FPS",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default=None,
        help="Extract target object using NLP and locate ONLY this target (e.g. 'phone', 'where is my bottle', 'laptop')",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable interactive GUI preview windows (useful for headless/automated runs)",
    )
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Run anchor-based relative spatial reasoning (e.g. 'bottle is left of chair')",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=25.0,
        help="Coordinate tolerance in pixels for horizontal/vertical alignment (default: 25.0)",
    )

    args = parser.parse_args()

    # Validate path arguments
    if args.source in ["image", "video"] and not args.path:
        parser.error(f"--path is required when --source is '{args.source}'")

    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    show_preview = not args.no_preview

    # Determine resolution
    if args.fast:
        imgsz = 480
    elif args.imgsz is not None:
        imgsz = args.imgsz
    else:
        imgsz = 512 if args.source == "webcam" else 640

    # 1. Device selection
    device = get_device(args.device)

    # 2. Ensure weights exist locally
    ensure_weights(model_path)

    # 3. Initialize detector and modular services
    detector = ObjectDetector(weights_path=model_path, device_preference=device)
    image_service = ImageService(detector)
    video_service = VideoService(detector)
    webcam_service = WebcamService(detector)

    # Extract target with NLP if provided
    target_ids = None
    target_names = None
    if args.target:
        nlp = get_nlp_extractor()
        nlp_res = nlp.extract(args.target)
        if nlp_res.target_classes:
            target_names = nlp_res.target_classes
            target_ids = set(nlp_res.target_class_ids)
        else:
            c_name = args.target.strip().lower()
            c_id = nlp.get_class_id(c_name)
            if c_id is not None:
                target_names = [nlp.id_to_class(c_id)]
                target_ids = {c_id}
            else:
                target_names = [c_name]
                target_ids = set()
        print(f"[NLP TARGET EXTRACTION] Target: {target_names} (COCO IDs: {list(target_ids)})")

    # 4. Dispatch mode
    if args.source == "image":
        run_image(
            image_service,
            args.path,
            args.conf,
            output_dir,
            show_preview,
            target_ids=target_ids,
            target_names=target_names,
            spatial=args.spatial,
            tolerance=args.tolerance,
        )
    elif args.source == "video":
        run_video(video_service, args.path, args.conf, output_dir, show_preview, target_ids=target_ids, target_names=target_names)
    elif args.source == "webcam":
        run_webcam(webcam_service, args.cam_id, args.conf, show_preview, imgsz=imgsz, target_names=target_names)
    elif args.source == "interactive":
        run_interactive_cli(webcam_service, detector, args.cam_id)


if __name__ == "__main__":
    main()
