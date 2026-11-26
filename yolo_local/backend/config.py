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
    default_model_name: str = "yolo11n.pt"
    default_model_path: Path = BASE_DIR / "models" / "yolo11n.pt"
    default_conf_threshold: float = 0.35
    default_imgsz: int = 640
    default_iou_threshold: float = 0.45
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
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_patience: float = 1.0
    whisper_no_speech_threshold: float = 0.4
    whisper_logprob_threshold: float = -0.8
    whisper_compression_ratio_threshold: float = 2.4
    whisper_condition_on_previous_text: bool = False
    whisper_preprocess_audio: bool = True
    whisper_normalize_gain: bool = True
    whisper_trim_silence: bool = True
    whisper_apply_preemphasis: bool = True
    whisper_initial_prompt: str = (
        "Hii, please find the bottle, cup, laptop, cell phone, chair, couch, "
        "dining table, microwave, oven, refrigerator, remote, hair drier, backpack, "
        "bag, wine glass, plate, bowl, book, clock, scissors, bed, potted plant, "
        "toothbrush, person, car, bicycle, motorcycle, airplane, bus, train, truck, "
        "boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, "
        "cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, umbrella, "
        "handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, "
        "baseball bat, baseball glove, skateboard, surfboard, tennis racket, fork, "
        "knife, spoon, banana, apple, sandwich, orange, broccoli, carrot, hot dog, "
        "pizza, donut, cake, toilet, mouse, keyboard, toaster, sink, vase, teddy bear. "
        "Where is the water bottle, cell phone, laptop, coffee mug, or chair?"
    )


settings = Settings()

# Ensure directories exist
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.whisper_models_dir.mkdir(parents=True, exist_ok=True)
settings.outputs_dir.mkdir(parents=True, exist_ok=True)
settings.static_dir.mkdir(parents=True, exist_ok=True)
