"""
Interactive Voice Assistant Engine.

Implements a real-time, stateful conversational loop for target object detection:
1. WAITING_FOR_WAKE_WORD: Spots "Hi Omii" (or phonetic variants).
2. ACKNOWLEDGING_WAKE_WORD: Speaks acknowledgment, mutes microphone to prevent acoustic feedback.
3. LISTENING_FOR_COMMAND: Records user's query with speech silence endpointing (VAD).
4. TRANSCRIBING: Uses local Whisper model to transcribe voice question.
5. EXTRACTING_TARGET: Extracts target COCO object and requested spatial region.
6. SEARCHING_FOR_OBJECT: Scans camera frames for up to 5.0 seconds (monotonic clock):
   - Exits immediately when the target matches the requested region.
   - If mismatched, continues checking remainder of 5 seconds before reporting mismatch.
   - Times out at 5.0s if object is absent.
7. GENERATING_RESPONSE: Builds natural-language spoken feedback using SpatialReasoner.
8. SPEAKING_RESPONSE: Delivers TTS feedback and transitions back to WAITING_FOR_WAKE_WORD.
9. ERROR_RECOVERY: Handles empty speech, unrecognized classes, or timeouts gracefully.
10. DISABLED: Idle state with zero CPU/GPU overhead when toggled OFF.
"""

from enum import Enum
import threading
import time
from typing import Any

from backend.schemas.interactive import (
    InteractiveCommandResponse,
    InteractiveStatusResponse,
)
from backend.services.detector import ObjectDetector, get_default_detector
from backend.services.nlp_extractor import NLPExtractor, get_nlp_extractor
from backend.services.spatial_reasoner import SpatialReasoner
from backend.services.stt_service import WhisperSTTService, get_stt_manager
from backend.services.webcam_service import WebcamService


class InteractiveState(str, Enum):
    """The 10 states of the interactive voice assistant state machine."""
    DISABLED = "DISABLED"
    WAITING_FOR_WAKE_WORD = "WAITING_FOR_WAKE_WORD"
    ACKNOWLEDGING_WAKE_WORD = "ACKNOWLEDGING_WAKE_WORD"
    LISTENING_FOR_COMMAND = "LISTENING_FOR_COMMAND"
    TRANSCRIBING = "TRANSCRIBING"
    EXTRACTING_TARGET = "EXTRACTING_TARGET"
    SEARCHING_FOR_OBJECT = "SEARCHING_FOR_OBJECT"
    GENERATING_RESPONSE = "GENERATING_RESPONSE"
    SPEAKING_RESPONSE = "SPEAKING_RESPONSE"
    ERROR_RECOVERY = "ERROR_RECOVERY"


# Standard wake phrases recognized by client spotter
WAKE_PHRASES = {
    "hi omii", "hi omi", "hey omii", "hey omi", "hi omee", "hey omee", "hello omii", "hello omi",
    "hi", "hii", "hello", "hey"
}

WAKE_ACKNOWLEDGMENT = "Yes, I'm listening. Which object should I find?"
UNCLEAR_AUDIO_MESSAGE = "Sorry, I couldn't understand that. Please try again."
UNKNOWN_TARGET_MESSAGE = "Sorry, I didn't recognize which object you asked for. Please ask again."


class InteractiveAssistant:
    """
    Coordinates state transitions, Whisper transcription, NLP target extraction,
    and 5-second YOLO search with spatial reasoning.
    """

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        nlp_extractor: NLPExtractor | None = None,
        webcam_service: WebcamService | None = None,
        stt_service: WhisperSTTService | None = None,
    ):
        self._detector = detector
        self._nlp = nlp_extractor
        self._webcam = webcam_service
        self._stt = stt_service

        self._active: bool = False
        self._state: InteractiveState = InteractiveState.DISABLED
        self._lock = threading.Lock()
        self._abort_event = threading.Event()

        # Telemetry & status variables
        self.active_target: str | None = None
        self.active_target_id: int | None = None
        self.requested_region: str | None = None
        self.last_speech: str | None = None
        self.last_response: str | None = None
        self.last_action_timestamp: float = time.time()

    # -----------------------------------------------------------------------
    # Lazy Service Properties (Reuses existing singleton models)
    # -----------------------------------------------------------------------
    @property
    def detector(self) -> ObjectDetector:
        if self._detector is None:
            self._detector = get_default_detector()
        return self._detector

    @property
    def nlp(self) -> NLPExtractor:
        if self._nlp is None:
            self._nlp = get_nlp_extractor()
        return self._nlp

    @property
    def webcam(self) -> WebcamService:
        if self._webcam is None:
            from backend.api.routes.stream import get_cam_service
            self._webcam = get_cam_service()
        return self._webcam

    @property
    def stt(self):
        if self._stt is None:
            self._stt = get_stt_manager()
        return self._stt

    @property
    def state(self) -> InteractiveState:
        with self._lock:
            return self._state

    def set_state(self, new_state: InteractiveState) -> None:
        with self._lock:
            self._state = new_state
            self.last_action_timestamp = time.time()

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    # -----------------------------------------------------------------------
    # Toggle Controls
    # -----------------------------------------------------------------------
    def enable(self) -> InteractiveStatusResponse:
        """Enables interactive mode. Transitions to WAITING_FOR_WAKE_WORD."""
        with self._lock:
            self._active = True
            self._abort_event.clear()
            self._state = InteractiveState.WAITING_FOR_WAKE_WORD
            self.last_action_timestamp = time.time()

        return self.get_status()

    def disable(self) -> InteractiveStatusResponse:
        """
        Disables interactive mode immediately.
        Signals abort to any running search loop and clears target lock.
        """
        with self._lock:
            self._active = False
            self._abort_event.set()
            self._state = InteractiveState.DISABLED
            self.active_target = None
            self.active_target_id = None
            self.requested_region = None
            self.last_action_timestamp = time.time()

        # Clear target filter in webcam service so manual view is restored
        try:
            self.webcam.clear_target_filter()
        except Exception:
            pass

        return self.get_status()

    def get_status(self) -> InteractiveStatusResponse:
        """Returns the current status snapshot of the interactive assistant."""
        with self._lock:
            return InteractiveStatusResponse(
                active=self._active,
                state=self._state.value,
                active_target=self.active_target,
                active_target_id=self.active_target_id,
                requested_region=self.requested_region,
                last_speech=self.last_speech,
                last_response=self.last_response,
                timestamp=self.last_action_timestamp,
            )

    # -----------------------------------------------------------------------
    # 5-Second Search Loop with Monotonic Clock
    # -----------------------------------------------------------------------
    def search_for_target_with_timeout(
        self,
        target_name: str,
        target_id: int,
        requested_region: str | None = None,
        timeout_seconds: float = 5.0,
        frame_override: Any = None,
    ) -> dict[str, Any]:
        """
        Polls camera frames for up to `timeout_seconds` using monotonic clock:
        - If target matches requested_region (or no region was requested), returns immediately!
        - If target is found but in wrong region, continues searching remainder of 5s.
        - If 5s expires without region match, returns the best mismatched candidate (if any) or not found.
        - Aborts immediately if disabled during execution.
        """
        start_time = time.monotonic()
        deadline = start_time + timeout_seconds

        best_mismatched_detection = None
        best_mismatched_spatial = None

        # Lock camera filter so live preview highlights the target exclusively
        self.webcam.set_target_filter([target_name])

        while time.monotonic() < deadline:
            # Check abort flag
            if self._abort_event.is_set() or not self._active:
                return {
                    "aborted": True,
                    "found": False,
                    "region_matched": False,
                    "detected_region": None,
                    "spatial_info": None,
                    "detection": None,
                    "detections": [],
                    "time_taken": round(time.monotonic() - start_time, 2),
                }

            # Retrieve frame
            if frame_override is not None:
                frame = frame_override
            else:
                frame = self.webcam.get_latest_frame()

            if frame is None:
                time.sleep(0.04)
                continue

            # Run YOLO detector exclusively for target class ID
            frame_h, frame_w = frame.shape[:2]
            detections, _, _ = self.detector.detect(
                frame,
                classes=[target_id],
                target_ids={target_id},
                conf_threshold=0.35,
                annotate=False,
            )

            if detections:
                # Sort detections by confidence descending
                detections.sort(key=lambda d: d.confidence, reverse=True)

                for det in detections:
                    spatial = SpatialReasoner.calculate_object_location(det.bbox, frame_w, frame_h)
                    detected_region = spatial["region"]

                    # Case A: No specific region requested by user -> immediate hit!
                    if not requested_region:
                        # Attempt to find anchor object in the same frame
                        rel_spatial = None
                        try:
                            rel_spatial = SpatialReasoner.locate_target_relative_to_anchor(
                                target_class=target_name,
                                frame=frame,
                                detector=self.detector,
                                conf_threshold=0.30,
                            )
                        except Exception:
                            rel_spatial = None

                        return {
                            "aborted": False,
                            "found": True,
                            "region_matched": True,
                            "detected_region": detected_region,
                            "spatial_info": spatial,
                            "detection": det,
                            "detections": [det.model_dump()],
                            "time_taken": round(time.monotonic() - start_time, 2),
                            "relative_spatial": rel_spatial,
                        }

                    # Case B: User requested a specific region
                    if SpatialReasoner.matches_requested_region(detected_region, requested_region):
                        # Matching region found -> exit search immediately!
                        rel_spatial = None
                        try:
                            rel_spatial = SpatialReasoner.locate_target_relative_to_anchor(
                                target_class=target_name,
                                frame=frame,
                                detector=self.detector,
                                conf_threshold=0.30,
                            )
                        except Exception:
                            rel_spatial = None

                        return {
                            "aborted": False,
                            "found": True,
                            "region_matched": True,
                            "detected_region": detected_region,
                            "spatial_info": spatial,
                            "detection": det,
                            "detections": [det.model_dump()],
                            "time_taken": round(time.monotonic() - start_time, 2),
                            "relative_spatial": rel_spatial,
                        }
                    else:
                        # Mismatched instance; keep best candidate and continue searching
                        if (
                            best_mismatched_detection is None
                            or det.confidence > best_mismatched_detection.confidence
                        ):
                            best_mismatched_detection = det
                            best_mismatched_spatial = spatial

            # If testing with a static frame override, don't spin indefinitely
            if frame_override is not None and (time.monotonic() - start_time > 0.15):
                break

            time.sleep(0.03)

        # Loop completed (5s timeout expired)
        total_time = round(time.monotonic() - start_time, 2)

        if best_mismatched_detection is not None and best_mismatched_spatial is not None:
            # Target was detected, but never in requested region
            return {
                "aborted": False,
                "found": True,
                "region_matched": False,
                "detected_region": best_mismatched_spatial["region"],
                "spatial_info": best_mismatched_spatial,
                "detection": best_mismatched_detection,
                "detections": [best_mismatched_detection.model_dump()],
                "time_taken": total_time,
            }

        # Target not detected anywhere in frame within 5 seconds
        return {
            "aborted": False,
            "found": False,
            "region_matched": False,
            "detected_region": None,
            "spatial_info": None,
            "detection": None,
            "detections": [],
            "time_taken": total_time,
        }

    # -----------------------------------------------------------------------
    # Command Execution (Voice Audio or Text)
    # -----------------------------------------------------------------------
    def process_text_command(
        self,
        query_text: str,
        frame_override: Any = None,
        timeout_seconds: float = 5.0,
    ) -> InteractiveCommandResponse:
        """
        Executes an interactive query from transcribed text or direct text input:
        1. EXTRACTING_TARGET: Identifies target object and requested region.
        2. SEARCHING_FOR_OBJECT: Runs 5-second YOLO search.
        3. GENERATING_RESPONSE: Builds natural language feedback.
        4. Returns to WAITING_FOR_WAKE_WORD (or DISABLED if toggled off).
        """
        clean_text = query_text.strip()
        self.last_speech = clean_text

        if not clean_text:
            self.set_state(InteractiveState.ERROR_RECOVERY)
            speech = UNCLEAR_AUDIO_MESSAGE
            self.last_response = speech
            self.set_state(
                InteractiveState.WAITING_FOR_WAKE_WORD
                if self.is_active()
                else InteractiveState.DISABLED
            )
            return InteractiveCommandResponse(
                success=False,
                state=self.state.value,
                transcribed_text="",
                target_object=None,
                target_class_id=None,
                requested_region=None,
                found=False,
                region_matched=False,
                detected_region=None,
                speech_response=speech,
                search_duration_seconds=0.0,
                detections=[],
            )

        # Check for simple wake greeting (e.g. "hi", "hii", "hello", "hi omii")
        lower_text = clean_text.lower().rstrip("!.,? ")
        if lower_text in WAKE_PHRASES:
            speech = WAKE_ACKNOWLEDGMENT
            self.last_response = speech
            self.set_state(
                InteractiveState.WAITING_FOR_WAKE_WORD
                if self.is_active()
                else InteractiveState.DISABLED
            )
            return InteractiveCommandResponse(
                success=True,
                state=self.state.value,
                transcribed_text=clean_text,
                target_object=None,
                target_class_id=None,
                requested_region=None,
                found=False,
                region_matched=False,
                detected_region=None,
                speech_response=speech,
                search_duration_seconds=0.0,
                detections=[],
            )

        # Step: Extract target & region
        self.set_state(InteractiveState.EXTRACTING_TARGET)
        extraction = self.nlp.extract_target_and_intent(clean_text)
        target_name = extraction["target_object"]
        target_id = extraction["target_class_id"]
        requested_region = extraction["requested_region"]

        self.active_target = target_name
        self.active_target_id = target_id
        self.requested_region = requested_region

        if not target_name or target_id is None:
            # Unrecognized object
            self.set_state(InteractiveState.ERROR_RECOVERY)
            speech = UNKNOWN_TARGET_MESSAGE
            self.last_response = speech
            self.set_state(
                InteractiveState.WAITING_FOR_WAKE_WORD
                if self.is_active()
                else InteractiveState.DISABLED
            )
            return InteractiveCommandResponse(
                success=False,
                state=self.state.value,
                transcribed_text=clean_text,
                target_object=None,
                target_class_id=None,
                requested_region=requested_region,
                found=False,
                region_matched=False,
                detected_region=None,
                speech_response=speech,
                search_duration_seconds=0.0,
                detections=[],
            )

        # Step: 5-Second Search
        self.set_state(InteractiveState.SEARCHING_FOR_OBJECT)
        search_res = self.search_for_target_with_timeout(
            target_name=target_name,
            target_id=target_id,
            requested_region=requested_region,
            timeout_seconds=timeout_seconds,
            frame_override=frame_override,
        )

        if search_res.get("aborted"):
            return InteractiveCommandResponse(
                success=False,
                state=InteractiveState.DISABLED.value,
                transcribed_text=clean_text,
                target_object=target_name,
                target_class_id=target_id,
                requested_region=requested_region,
                found=False,
                region_matched=False,
                detected_region=None,
                speech_response="Interactive mode was cancelled.",
                search_duration_seconds=search_res["time_taken"],
                detections=[],
            )

        # Step: Generating response
        self.set_state(InteractiveState.GENERATING_RESPONSE)
        rel_spatial = search_res.get("relative_spatial")

        anchor_name = None
        rel_direction = None
        rel_dict = None

        if rel_spatial is not None:
            rel_dict = rel_spatial.to_dict()
            if rel_spatial.anchor:
                anchor_name = rel_spatial.anchor.class_name
                rel_direction = rel_spatial.direction
                # Use natural language anchor relative description
                speech = rel_spatial.description
            else:
                speech = SpatialReasoner.generate_location_response(
                    target_object=target_name,
                    found=search_res["found"],
                    detected_region=search_res["detected_region"],
                    requested_region=requested_region,
                    region_matched=search_res["region_matched"],
                    timed_out=not search_res["found"],
                )
        else:
            speech = SpatialReasoner.generate_location_response(
                target_object=target_name,
                found=search_res["found"],
                detected_region=search_res["detected_region"],
                requested_region=requested_region,
                region_matched=search_res["region_matched"],
                timed_out=not search_res["found"],
            )

        self.last_response = speech

        # Return to waiting state if still active
        self.set_state(
            InteractiveState.WAITING_FOR_WAKE_WORD
            if self.is_active()
            else InteractiveState.DISABLED
        )

        return InteractiveCommandResponse(
            success=True,
            state=self.state.value,
            transcribed_text=clean_text,
            target_object=target_name,
            target_class_id=target_id,
            requested_region=requested_region,
            found=search_res["found"],
            region_matched=search_res["region_matched"],
            detected_region=search_res["detected_region"],
            speech_response=speech,
            search_duration_seconds=search_res["time_taken"],
            detections=search_res["detections"],
            anchor_object=anchor_name,
            relative_direction=rel_direction,
            relative_spatial=rel_dict,
        )

    def process_audio_command(
        self,
        audio_bytes: bytes,
        filename_hint: str = "voice_command.wav",
        frame_override: Any = None,
        timeout_seconds: float = 5.0,
        hint_text: str | None = None,
    ) -> InteractiveCommandResponse:
        """
        Transcribes audio using local Whisper, then runs full target localization.
        If hint_text is provided by browser speech recognition or client fallback,
        it is fused to ensure accurate object identification even if microphone audio
        is noisy or Whisper mishears.
        """
        if not audio_bytes or len(audio_bytes) < 100:
            query = (hint_text or "").strip()
            return self.process_text_command(query, frame_override, timeout_seconds)

        self.set_state(InteractiveState.TRANSCRIBING)
        try:
            if hasattr(self.stt, "transcribe_with_hint"):
                fused_result = self.stt.transcribe_with_hint(audio_bytes, hint_text=hint_text)
                chosen_text = fused_result.get("text", "").strip()
            else:
                transcribed_text = self.stt.transcribe(audio_bytes).strip()
                chosen_text = transcribed_text
                if hint_text and hint_text.strip():
                    hint_clean = hint_text.strip()
                    if not transcribed_text:
                        chosen_text = hint_clean
                    else:
                        extraction_whisper = self.nlp.extract_target_and_intent(transcribed_text)
                        if not extraction_whisper.get("valid"):
                            extraction_hint = self.nlp.extract_target_and_intent(hint_clean)
                            if extraction_hint.get("valid"):
                                chosen_text = hint_clean
        except Exception as exc:
            print(f"[ERROR] Whisper transcription failed: {exc}")
            chosen_text = (hint_text or "").strip()

        return self.process_text_command(
            query_text=chosen_text,
            frame_override=frame_override,
            timeout_seconds=timeout_seconds,
        )


_DEFAULT_INTERACTIVE_ASSISTANT: InteractiveAssistant | None = None


def get_interactive_assistant() -> InteractiveAssistant:
    """Singleton accessor for the InteractiveAssistant."""
    global _DEFAULT_INTERACTIVE_ASSISTANT
    if _DEFAULT_INTERACTIVE_ASSISTANT is None:
        _DEFAULT_INTERACTIVE_ASSISTANT = InteractiveAssistant()
    return _DEFAULT_INTERACTIVE_ASSISTANT
