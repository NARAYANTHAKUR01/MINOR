from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")
    width: float = Field(..., description="Box width in pixels")
    height: float = Field(..., description="Box height in pixels")


class DetectionItem(BaseModel):
    class_id: int = Field(..., description="COCO Class ID")
    class_name: str = Field(..., description="Name of the detected class (e.g., person, car)")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")


class DetectionResultResponse(BaseModel):
    success: bool = True
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    num_detections: int = Field(..., description="Total count of detected objects")
    device_used: str = Field(..., description="Device used for inference (MPS or CPU)")
    detections: list[DetectionItem] = Field(default_factory=list, description="List of detected objects")
    annotated_image_url: str | None = Field(default=None, description="URL or relative path to annotated image")


class DeviceInfoResponse(BaseModel):
    device: str
    device_name: str
    mps_available: bool
    mps_built: bool
    torch_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    weights_path: str
    device: str
    total_classes: int
    classes: dict[int, str]
