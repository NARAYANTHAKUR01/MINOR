import os
from pathlib import Path
import shutil
import sys
from ultralytics import YOLO

from backend.config import settings
from backend.core.device import get_device

# In-memory cache for loaded YOLO model instances
_LOADED_MODELS: dict[str, YOLO] = {}


def ensure_weights(weights_path: Path | None = None) -> Path:
    """
    Verifies that the weights file exists locally.
    If it exists, loads locally without downloading.
    If not, downloads once and copies to the target path.
    """
    target_path = Path(weights_path) if weights_path else settings.default_model_path

    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[INFO] Using existing local weights: {target_path}")
        return target_path

    # Fallback to alternate local weight names if present
    for alt_name in ("yolo11n.pt", "yolo11m.pt", "YOLOv11n.pt", "YOLOv11m.pt"):
        alt_path = target_path.parent / alt_name
        if alt_path.exists() and alt_path.stat().st_size > 0:
            print(f"[INFO] Using local weights: {alt_path}")
            return alt_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Local model weights not found at '{target_path}'.")
    print(f"[INFO] Downloading YOLO11m weights once into '{target_path}'...")

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded_file = attempt_download_asset("yolo11m.pt")
        downloaded_path = Path(downloaded_file).resolve()

        if downloaded_path != target_path.resolve():
            shutil.copy2(downloaded_path, target_path)

        print(f"[SUCCESS] Weights downloaded and saved to: {target_path}")
        return target_path
    except Exception as exc:
        print(f"[ERROR] Failed to download model weights: {exc}", file=sys.stderr)
        raise RuntimeError(f"Could not prepare model weights: {exc}") from exc


def get_yolo_model(weights_path: Path | None = None, requested_device: str = "auto") -> tuple[YOLO, str]:
    """
    Get or load a cached YOLO model instance.
    """
    model_path = ensure_weights(weights_path)
    device = get_device(requested_device)
    cache_key = f"{model_path.resolve()}_{device}"

    if cache_key in _LOADED_MODELS:
        return _LOADED_MODELS[cache_key], device

    print(f"[INFO] Loading YOLO11m into memory from '{model_path}' on device '{device}'...")
    model = YOLO(str(model_path))
    _LOADED_MODELS[cache_key] = model
    return model, device
