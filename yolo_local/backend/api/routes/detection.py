from pathlib import Path
import shutil
import uuid
import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from backend.config import settings
from backend.schemas.detection import DetectionResultResponse
from backend.services.image_service import ImageService
from backend.services.video_service import VideoService

router = APIRouter(prefix="/detection", tags=["Detection"])

# Lazy service instances
_image_service: ImageService | None = None
_video_service: VideoService | None = None


def get_img_service() -> ImageService:
    global _image_service
    if _image_service is None:
        _image_service = ImageService()
    return _image_service


def get_vid_service() -> VideoService:
    global _video_service
    if _video_service is None:
        _video_service = VideoService()
    return _video_service


@router.post("/image", response_model=DetectionResultResponse)
async def detect_image(
    file: UploadFile = File(..., description="Image file to detect objects in"),
    conf: float = Query(0.4, ge=0.01, le=1.0, description="Confidence threshold"),
):
    """
    Upload an image file and receive structured detections (bounding boxes, classes, confidences)
    and an annotated image URL.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    contents = await file.read()
    try:
        response, _, _ = get_img_service().process_image_bytes(
            image_bytes=contents,
            filename=file.filename,
            conf_threshold=conf,
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {exc}")


@router.post("/image/direct")
async def detect_image_direct(
    file: UploadFile = File(..., description="Image file"),
    conf: float = Query(0.4, ge=0.01, le=1.0, description="Confidence threshold"),
):
    """
    Upload an image and get the annotated JPEG image back directly in the HTTP response.
    Ideal for direct display in frontend <img> or canvas elements.
    """
    contents = await file.read()
    try:
        _, annotated, _ = get_img_service().process_image_bytes(
            image_bytes=contents,
            filename=file.filename or "upload.jpg",
            conf_threshold=conf,
        )
        success, buffer = cv2.imencode(".jpg", annotated)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode annotated image.")
        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/video")
async def detect_video(
    file: UploadFile = File(..., description="Video file to annotate"),
    conf: float = Query(0.4, ge=0.01, le=1.0, description="Confidence threshold"),
):
    """
    Upload a video file for frame-by-frame object detection.
    Returns the URL to the exported annotated MP4 video.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    # Save uploaded video temporarily
    temp_in = settings.outputs_dir / f"upload_{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(temp_in, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        out_video_path = get_vid_service().process_video_file(
            video_path=temp_in,
            conf_threshold=conf,
            show_preview=False,
        )
        return {
            "success": True,
            "filename": out_video_path.name,
            "video_url": f"/outputs/{out_video_path.name}",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}")
    finally:
        if temp_in.exists():
            temp_in.unlink()


@router.post("/spatial")
async def detect_spatial_reasoning(
    file: UploadFile = File(..., description="Image file to analyze"),
    target: str = Form(..., description="Target object class name or voice query (e.g. 'bottle' or 'where is the bottle')"),
    conf: float = Form(0.35, ge=0.01, le=1.0, description="Confidence threshold"),
    tolerance: float = Form(25.0, ge=1.0, le=200.0, description="Spatial alignment tolerance in pixels"),
):
    """
    Analyzes spatial relationships between the target object and reference/anchor objects in an image.
    Returns natural language spatial feedback (e.g. 'The bottle is to the left of the chair.').
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    # Extract target class from query if given natural sentence
    from backend.services.nlp_extractor import get_nlp_extractor
    nlp = get_nlp_extractor()
    nlp_res = nlp.extract(target)
    target_class = nlp_res.target_classes[0] if nlp_res.target_classes else target.strip().lower()

    temp_in = settings.outputs_dir / f"upload_spatial_{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(temp_in, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        spatial_res, annotated, saved_path = get_img_service().process_image_spatial_reasoning(
            image_path=temp_in,
            target_class=target_class,
            conf_threshold=conf,
            tolerance_x=tolerance,
            tolerance_y=tolerance,
            save_output=True,
        )
        return {
            "success": True,
            "target_class": target_class,
            "description": spatial_res.description,
            "spatial_result": spatial_res.to_dict(),
            "annotated_image_url": f"/outputs/{saved_path.name}" if saved_path else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Spatial reasoning failed: {exc}")
    finally:
        if temp_in.exists():
            temp_in.unlink()
