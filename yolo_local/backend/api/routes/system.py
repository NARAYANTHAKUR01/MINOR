from fastapi import APIRouter
from backend.core.device import get_device_info
from backend.core.model_loader import get_yolo_model
from backend.schemas.detection import DeviceInfoResponse, ModelInfoResponse
from backend.config import settings

router = APIRouter(prefix="/system", tags=["System"])


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "yolo_local_backend"}


@router.get("/device", response_model=DeviceInfoResponse)
def get_device_status():
    """Returns hardware acceleration details (Apple Silicon MPS / CPU)."""
    return get_device_info()


@router.get("/model", response_model=ModelInfoResponse)
def get_model_status():
    """Returns YOLO11m model metadata and classes."""
    model, device = get_yolo_model()
    return ModelInfoResponse(
        model_name="YOLO11m",
        weights_path=str(settings.default_model_path),
        device=device.upper(),
        total_classes=len(model.names),
        classes=model.names,
    )
