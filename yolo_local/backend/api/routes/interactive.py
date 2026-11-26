"""
FastAPI route endpoints for Interactive Voice Assistant mode.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend.schemas.interactive import (
    InteractiveCommandRequest,
    InteractiveCommandResponse,
    InteractiveStatusResponse,
    InteractiveToggleRequest,
)
from backend.services.interactive_assistant import (
    InteractiveState,
    get_interactive_assistant,
)

router = APIRouter(prefix="/interactive", tags=["Interactive Mode"])


class SetStateRequest(BaseModel):
    state: str


@router.post("/toggle", response_model=InteractiveStatusResponse)
def toggle_interactive_mode(req: InteractiveToggleRequest):
    """
    Toggles Interactive Assistant mode ON or OFF:
    - When OFF: Mic & spotter inactive, 0% CPU/GPU idle load, standard detection works.
    - When ON: Begins waiting for wake phrase 'Hi Omii'.
    """
    assistant = get_interactive_assistant()
    if req.enable:
        return assistant.enable()
    else:
        return assistant.disable()


@router.get("/status", response_model=InteractiveStatusResponse)
def get_interactive_status():
    """Returns the current state machine status and active target metadata."""
    assistant = get_interactive_assistant()
    return assistant.get_status()


@router.post("/set-state", response_model=InteractiveStatusResponse)
def update_interactive_state(req: SetStateRequest):
    """Updates the state machine state from the client-side lifecycle."""
    assistant = get_interactive_assistant()
    try:
        new_state = InteractiveState(req.state)
        assistant.set_state(new_state)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid state: {req.state}")
    return assistant.get_status()


@router.post("/process-audio", response_model=InteractiveCommandResponse)
async def process_audio_command(
    audio: UploadFile = File(...),
    hint_text: str | None = Form(None),
):
    """
    Processes a recorded voice question:
    1. Transcribes speech using local Whisper.
    2. Extracts target COCO class & requested spatial region.
    3. Searches camera frames for up to 5 seconds.
    4. Returns spatial feedback & spoken sentence for TTS.
    """
    assistant = get_interactive_assistant()
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    return assistant.process_audio_command(
        audio_bytes=audio_bytes,
        filename_hint=audio.filename or "question.wav",
        hint_text=hint_text,
    )


@router.post("/text-command", response_model=InteractiveCommandResponse)
def process_text_command(req: InteractiveCommandRequest):
    """
    Direct text query runner for testing, simulation, or fallback keyboard input.
    """
    assistant = get_interactive_assistant()
    return assistant.process_text_command(req.text)
