"""
Pydantic schemas for the Interactive Voice Assistant mode.
"""

from typing import Any
from pydantic import BaseModel, Field


class InteractiveToggleRequest(BaseModel):
    enable: bool = Field(..., description="True to enable interactive mode, False to disable.")


class InteractiveCommandRequest(BaseModel):
    text: str = Field(..., description="Text query to simulate or run directly in interactive mode.")


class InteractiveStatusResponse(BaseModel):
    active: bool = Field(..., description="Whether interactive mode is enabled.")
    state: str = Field(..., description="Current state machine enum value.")
    active_target: str | None = Field(default=None, description="Current or last target object queried.")
    active_target_id: int | None = Field(default=None, description="COCO class ID for active target.")
    requested_region: str | None = Field(default=None, description="Requested spatial region if any.")
    last_speech: str | None = Field(default=None, description="Last transcribed speech.")
    last_response: str | None = Field(default=None, description="Last generated response spoken to the user.")
    timestamp: float = Field(..., description="Unix timestamp.")


class InteractiveCommandResponse(BaseModel):
    success: bool = Field(..., description="Whether the interaction completed successfully.")
    state: str = Field(..., description="State after execution.")
    transcribed_text: str = Field(..., description="Transcribed query or provided text.")
    target_object: str | None = Field(default=None, description="Extracted target object class name.")
    target_class_id: int | None = Field(default=None, description="Extracted target object class ID.")
    requested_region: str | None = Field(default=None, description="Requested spatial direction if any.")
    found: bool = Field(..., description="Whether the target object was detected in camera view.")
    region_matched: bool = Field(..., description="Whether the detected object matched requested region.")
    detected_region: str | None = Field(default=None, description="Detected 3x3 region in the frame.")
    speech_response: str = Field(..., description="Natural language spoken response generated.")
    search_duration_seconds: float = Field(..., description="Seconds spent searching camera frames.")
    detections: list[dict[str, Any]] = Field(default_factory=list, description="Telemetry for detected objects.")
