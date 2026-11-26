"""
Backend service layer package exports.
Uses PEP 562 module lazy-loading for lightning-fast CLI and script startup.
"""

from typing import Any

__all__ = [
    "ObjectDetector",
    "get_default_detector",
    "ImageService",
    "VideoService",
    "Visualizer",
    "WebcamService",
    "NLPExtractor",
    "get_nlp_extractor",
    "BaseSTTService",
    "MockSTTService",
    "WhisperSTTService",
    "STTManager",
    "get_stt_manager",
    "SpatialReasoner",
    "InteractiveAssistant",
    "InteractiveState",
    "get_interactive_assistant",
]


def __getattr__(name: str) -> Any:
    if name in ("ObjectDetector", "get_default_detector"):
        from backend.services import detector
        return getattr(detector, name)
    if name == "ImageService":
        from backend.services.image_service import ImageService
        return ImageService
    if name == "VideoService":
        from backend.services.video_service import VideoService
        return VideoService
    if name == "Visualizer":
        from backend.services.visualizer import Visualizer
        return Visualizer
    if name == "WebcamService":
        from backend.services.webcam_service import WebcamService
        return WebcamService
    if name in ("NLPExtractor", "get_nlp_extractor"):
        from backend.services import nlp_extractor
        return getattr(nlp_extractor, name)
    if name in (
        "BaseSTTService",
        "MockSTTService",
        "WhisperSTTService",
        "STTManager",
        "get_stt_manager",
    ):
        from backend.services import stt_service
        return getattr(stt_service, name)
    if name == "SpatialReasoner":
        from backend.services.spatial_reasoner import SpatialReasoner
        return SpatialReasoner
    if name in ("InteractiveAssistant", "InteractiveState", "get_interactive_assistant"):
        from backend.services import interactive_assistant
        return getattr(interactive_assistant, name)
        from backend.services import stt_service
        return getattr(stt_service, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
