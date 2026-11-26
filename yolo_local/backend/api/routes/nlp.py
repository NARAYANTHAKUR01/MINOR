"""
FastAPI Router for Speech-to-Text & NLP Object Extraction.
Provides endpoints for extracting objects from spoken sentences,
transcribing audio locally using open-source Whisper, filtering YOLO
detections based on voice queries, and managing STT providers.
"""

from pathlib import Path
import time
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.config import settings
from backend.schemas.detection import DetectionItem, DetectionResultResponse
from backend.schemas.nlp import (
    ExtractSentenceRequest,
    NLPExtractionResponse,
    STTStatusResponse,
    SupportedClassesResponse,
    TranscribeAndExtractResponse,
    TranscribeAudioResponse,
)
from backend.services.detector import get_default_detector
from backend.services.nlp_extractor import get_nlp_extractor
from backend.services.stt_service import get_stt_manager

router = APIRouter(prefix="/nlp", tags=["NLP Object Extraction & Voice"])


@router.get("/stt-status", response_model=STTStatusResponse)
def get_stt_status():
    """
    Returns the operational status of the local Speech-to-Text engine,
    including active provider, device, local model weights path, and availability.
    """
    stt = get_stt_manager()
    status = stt.get_status()
    return STTStatusResponse(**status)


@router.post("/transcribe", response_model=TranscribeAudioResponse)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Speech audio file (.wav, .mp3, .m4a, .webm, .ogg)"),
    provider: str | None = Form(None, description="STT provider ('whisper', 'mock', etc.)"),
):
    """
    Transcribe speech audio into text using the open-source Whisper model running 100% locally.
    Supports audio recordings (.webm, .wav, .mp3, .ogg, .m4a) from browser microphone or local files.
    """
    stt = get_stt_manager()
    try:
        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

        detailed = stt.transcribe_detailed(audio_bytes, provider_name=provider)
        return TranscribeAudioResponse(
            success=True,
            transcription=detailed["text"],
            provider=detailed["model"],
            model_size=detailed.get("model_size", "base"),
            processing_time_ms=detailed["processing_time_ms"],
            device=detailed.get("device", "cpu"),
            language=detailed.get("language"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {exc}")


@router.post("/extract", response_model=NLPExtractionResponse)
def extract_objects_from_sentence(request: ExtractSentenceRequest):
    """
    Extract COCO object classes from a spoken sentence or text query using NLP.
    
    Supports:
    - 80 COCO classes with exact ID mappings (0 to 79)
    - Longest-phrase matching first (e.g. 'hot dog' vs 'dog', 'teddy bear' vs 'bear')
    - Plural handling and lemmatization (dogs -> dog, knives -> knife, people -> person)
    - Colloquial aliases & synonyms (bike -> bicycle, phone -> cell phone, sofa -> couch)
    - Negation detection ('find the person, not the car')
    - Intent detection ('locate', 'count', 'detect', 'filter', 'general')
    - Modifiers and quantities ('two red apples')
    """
    nlp = get_nlp_extractor()
    return nlp.extract(
        sentence=request.sentence,
        include_attributes=request.include_attributes,
        allow_fuzzy=request.allow_fuzzy,
        filter_negated=request.filter_negated,
    )


@router.post("/transcribe-and-extract", response_model=TranscribeAndExtractResponse)
async def transcribe_and_extract(
    audio_file: UploadFile | None = File(None, description="Optional audio file (.wav, .mp3, .m4a, .webm)"),
    sentence_text: str | None = Form(None, description="Direct sentence string if already transcribed"),
    provider: str | None = Form(None, description="STT provider to use ('whisper', 'mock', etc.)"),
):
    """
    Transcribe speech audio (via the local open-source Whisper model) and extract COCO objects via NLP.
    Accepts either an uploaded audio recording or an existing sentence string.
    """
    stt = get_stt_manager()

    try:
        if sentence_text and sentence_text.strip():
            # Direct text provided
            sentence = sentence_text.strip()
            used_provider = provider or "text_input"
            nlp_res = stt.nlp.extract(sentence)
        elif audio_file:
            # Audio file uploaded
            audio_bytes = await audio_file.read()
            sentence, used_provider, nlp_res = stt.transcribe_and_extract(
                audio_data=audio_bytes,
                provider_name=provider,
            )
        else:
            # Fallback to active provider's default test sentence
            sentence, used_provider, nlp_res = stt.transcribe_and_extract(
                audio_data="",
                provider_name=provider,
            )

        return TranscribeAndExtractResponse(
            success=True,
            transcribed_sentence=sentence,
            stt_provider=used_provider,
            nlp_result=nlp_res,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {exc}")


@router.post("/query-image")
async def query_image_with_sentence(
    image: UploadFile = File(..., description="Image file to detect objects in"),
    sentence: str = Form(..., description="Sentence query (e.g. 'Can you spot the person and dog?')"),
    conf: float = Form(0.35, description="YOLO confidence threshold"),
):
    """
    End-to-End Voice/Sentence Visual Search:
    1. Extracts target objects from the sentence using NLP.
    2. Runs YOLO object detection on the provided image.
    3. Filters detections to ONLY the objects requested in the sentence.
    4. Annotates the image and returns structured detection bounding boxes.
    """
    nlp = get_nlp_extractor()
    nlp_res = nlp.extract(sentence)

    # Read image bytes
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    detector = get_default_detector()
    start_t = time.perf_counter()
    all_detections, _, _ = detector.detect(
        frame=frame,
        conf_threshold=conf,
        annotate=False,
    )

    # Filter detections to target classes extracted by NLP
    target_ids = set(nlp_res.target_class_ids)
    if target_ids:
        matched_detections = [d for d in all_detections if d.class_id in target_ids]
    else:
        # If no specific COCO targets recognized in query, return all detections
        matched_detections = all_detections

    elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

    # Generate annotated image highlighting the matched targets
    annotated_frame = frame.copy()
    for d in matched_detections:
        p1 = (int(d.bbox.x1), int(d.bbox.y1))
        p2 = (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(annotated_frame, p1, p2, (0, 255, 128), 2)
        label = f"{d.class_name} {int(d.confidence * 100)}%"
        cv2.putText(
            annotated_frame,
            label,
            (p1[0], max(p1[1] - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 128),
            2,
        )

    # Save output image
    filename = f"nlp_query_{uuid.uuid4().hex[:8]}.jpg"
    out_path = settings.outputs_dir / filename
    cv2.imwrite(str(out_path), annotated_frame)

    return {
        "success": True,
        "sentence": sentence,
        "nlp_analysis": nlp_res,
        "yolo_results": {
            "inference_time_ms": elapsed_ms,
            "target_classes_searched": nlp_res.target_classes,
            "total_detected": len(matched_detections),
            "detections": matched_detections,
            "annotated_image_url": f"/outputs/{filename}",
        },
    }


@router.post("/query-image-audio")
async def query_image_with_audio(
    image: UploadFile = File(..., description="Image file to detect objects in"),
    audio: UploadFile = File(..., description="Audio recording with spoken voice query"),
    conf: float = Form(0.35, description="YOLO confidence threshold"),
):
    """
    End-to-End Voice Audio Visual Search:
    1. Transcribes voice audio locally using open-source Whisper STT.
    2. Extracts target objects from the transcription using NLP.
    3. Runs YOLO object detection on the image.
    4. Filters detections to ONLY the objects requested in the spoken audio.
    5. Annotates the image and returns detection results.
    """
    stt = get_stt_manager()
    audio_bytes = await audio.read()
    detailed_stt = stt.transcribe_detailed(audio_bytes)
    sentence = detailed_stt["text"]

    nlp = get_nlp_extractor()
    nlp_res = nlp.extract(sentence)

    # Read image bytes
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    detector = get_default_detector()
    start_t = time.perf_counter()
    all_detections, _, _ = detector.detect(
        frame=frame,
        conf_threshold=conf,
        annotate=False,
    )

    target_ids = set(nlp_res.target_class_ids)
    if target_ids:
        matched_detections = [d for d in all_detections if d.class_id in target_ids]
    else:
        matched_detections = all_detections

    elapsed_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

    annotated_frame = frame.copy()
    for d in matched_detections:
        p1 = (int(d.bbox.x1), int(d.bbox.y1))
        p2 = (int(d.bbox.x2), int(d.bbox.y2))
        cv2.rectangle(annotated_frame, p1, p2, (0, 255, 128), 2)
        label = f"{d.class_name} {int(d.confidence * 100)}%"
        cv2.putText(
            annotated_frame,
            label,
            (p1[0], max(p1[1] - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 128),
            2,
        )

    filename = f"nlp_audio_query_{uuid.uuid4().hex[:8]}.jpg"
    out_path = settings.outputs_dir / filename
    cv2.imwrite(str(out_path), annotated_frame)

    return {
        "success": True,
        "transcription": sentence,
        "stt_metadata": detailed_stt,
        "nlp_analysis": nlp_res,
        "yolo_results": {
            "inference_time_ms": elapsed_ms,
            "target_classes_searched": nlp_res.target_classes,
            "total_detected": len(matched_detections),
            "detections": matched_detections,
            "annotated_image_url": f"/outputs/{filename}",
        },
    }


@router.get("/classes", response_model=SupportedClassesResponse)
def get_supported_classes():
    """
    Returns the full list of all 80 COCO classes, their IDs,
    common spoken synonyms/aliases, and prompt examples.
    """
    nlp = get_nlp_extractor()
    return nlp.get_supported_classes()


@router.get("/stt-providers")
def get_stt_providers():
    """Returns available and currently active Speech-To-Text providers."""
    stt = get_stt_manager()
    return {
        "active_provider": stt.active_provider.provider_name,
        "available_providers": stt.get_available_providers(),
    }


@router.post("/set-stt-provider")
def set_stt_provider(provider_name: str = Query(..., description="Provider name ('whisper', 'mock', etc.)")):
    """Switch the active STT provider."""
    stt = get_stt_manager()
    try:
        stt.set_active_provider(provider_name)
        return {
            "success": True,
            "active_provider": stt.active_provider.provider_name,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
