import os
from pathlib import Path
from pydantic import BaseModel

# Base project directory (yolo_local root)
BASE_DIR = Path(__file__).resolve().parent.parent

# Set YOLO config directory locally inside the project to prevent macOS permission warnings
os.environ["YOLO_CONFIG_DIR"] = str(BASE_DIR / ".ultralytics")


class Settings(BaseModel):
    """Global configuration settings for YOLO local backend."""

    # Project Paths
    base_dir: Path = BASE_DIR
    models_dir: Path = BASE_DIR / "models"
    outputs_dir: Path = BASE_DIR / "outputs"
    static_dir: Path = BASE_DIR / "static"

    # Model settings
    default_model_name: str = "yolo11m.pt"
    default_model_path: Path = BASE_DIR / "models" / "yolo11m.pt"
    default_conf_threshold: float = 0.4
    weights_download_url: str = (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
    )

    # API Server settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    # Webcam defaults
    default_cam_id: int = 0

    # Local Speech-to-Text settings (Open-Source Whisper)
    stt_provider: str = "whisper"
    stt_model_name: str = "base"
    whisper_models_dir: Path = BASE_DIR / "models" / "whisper"
    stt_device: str = "cpu"
    whisper_language: str = "en"
    whisper_temperature: float = 0.0
    whisper_initial_prompt: str = (
        "Hi Omii, please find the person, car, bicycle, dog, cat, bottle, cup, laptop, "
        "cell phone, microwave, oven, refrigerator, remote, hair drier, chair, couch, "
        "table, bag, backpack, glass, plate, bowl, book, clock, scissors, bed, plant, toothbrush."
    )


settings = Settings()

# Ensure directories exist
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.whisper_models_dir.mkdir(parents=True, exist_ok=True)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
settings.static_dir.mkdir(parents=True, exist_ok=True)
