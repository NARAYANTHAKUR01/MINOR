from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from backend.services.stt_service import get_stt_manager
from backend.services.webcam_service import WebcamService

router = APIRouter(prefix="/stream", tags=["Streaming"])

_webcam_service: WebcamService | None = None


def get_cam_service() -> WebcamService:
    global _webcam_service
    if _webcam_service is None:
        _webcam_service = WebcamService()
    return _webcam_service


@router.get("/webcam")
def stream_webcam(
    cam_id: int = Query(0, description="Webcam device index"),
    conf: float = Query(0.4, ge=0.01, le=1.0, description="Confidence threshold"),
    imgsz: int = Query(512, description="Inference image resolution (default 512 for 35-45 FPS)"),
):
    """
    Live real-time MJPEG camera stream with YOLO11m detections and FPS counter.
    Compatible directly with standard HTML: `<img src="/api/stream/webcam" />`.
    """
    try:
        frame_generator = get_cam_service().generate_mjpeg_stream(
            cam_id=cam_id,
            conf_threshold=conf,
            imgsz=imgsz,
        )
        return StreamingResponse(
            frame_generator,
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Camera streaming failed. Ensure macOS camera permission is enabled for Terminal/IDE: {exc}",
        )


@router.get("/telemetry")
def get_telemetry():
    """Returns real-time spatial telemetry of detected objects (azimuth, sector, range)."""
    return get_cam_service().get_latest_telemetry()


@router.get("/target")
def get_target_status():
    """Returns the current exclusive target lock status."""
    return get_cam_service().get_target_status()


@router.post("/set-target")
def set_target_filter(body: dict):
    """
    Sets exclusive target lock to locate ONLY requested object(s) extracted from query text or target name.
    Body format: {"target": "bottle"} or {"query": "where is my cell phone?"}
    """
    target_input = body.get("target") or body.get("query") or body.get("sentence", "")
    if not target_input:
        raise HTTPException(status_code=400, detail="Missing 'target' or 'query' field.")
    return get_cam_service().set_target_filter(target_input)


@router.post("/clear-target")
def clear_target_filter():
    """Clears exclusive target lock to restore full detection across all classes."""
    return get_cam_service().clear_target_filter()


@router.post("/query")
def query_spatial_object(body: dict):
    """
    Matches voice or text query with live detected objects in the environment
    and returns exact spatial localization and synthesized feedback text.
    """
    query_text = body.get("query", "").lower()
    return get_cam_service().query_target(query_text)


@router.post("/query-audio")
async def query_spatial_object_audio(
    audio_file: UploadFile = File(..., description="Audio recording of spoken command"),
    hint_text: str | None = Form(None, description="Optional client-side speech hint"),
):
    """
    Transcribes live audio recording from microphone using local Whisper STT,
    extracts objects with NLP, matches against current live camera telemetry,
    and returns spatial coordinates and spoken response.
    """
    stt = get_stt_manager()
    try:
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio recording.")

        detailed_stt = stt.transcribe_detailed(audio_bytes)
        sentence = detailed_stt.get("text", "").strip()

        if hint_text and hint_text.strip():
            hint_clean = hint_text.strip()
            if not sentence:
                sentence = hint_clean
            else:
                nlp = stt.nlp
                if not nlp.extract(sentence).target_classes and nlp.extract(hint_clean).target_classes:
                    sentence = hint_clean

        result = get_cam_service().query_target(sentence)
        result["transcribed_sentence"] = sentence
        result["stt_metadata"] = detailed_stt
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice query processing failed: {exc}")
