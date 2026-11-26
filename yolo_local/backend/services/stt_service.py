"""
Pluggable Speech-to-Text (STT) Service Interface.

Provides a clean abstraction for integrating Speech-to-Text models.
Included providers:
1. WhisperSTTService: Local, offline OpenAI Whisper STT running on Apple Silicon / CPU.
2. MockSTTService: Built-in development & testing provider.
3. CustomSTTService: Pluggable slot for any external/user-provided STT model.
"""

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any

import numpy as np

from backend.config import settings
from backend.schemas.nlp import NLPExtractionResponse
from backend.services.nlp_extractor import NLPExtractor, get_nlp_extractor

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    High-performance audio preprocessing pipeline for Speech-to-Text:
    1. DC offset removal (zero-centering waveform)
    2. Dynamic energy-based silence trimming (removing leading/trailing dead air)
    3. Peak amplitude & RMS normalization (scaling to 0.95 peak to maximize SNR)
    4. Duration safeguarding (padding sub-0.4s audio to prevent encoder artifacts)
    """

    TARGET_PEAK: float = 0.95
    SAMPLE_RATE: int = 16000
    MIN_DURATION_SECONDS: float = 0.4
    SILENCE_THRESHOLD_RATIO: float = 0.04
    PADDING_DURATION_SECONDS: float = 0.08

    @classmethod
    def preprocess_waveform(
        cls,
        audio: np.ndarray,
        sr: int = 16000,
        trim_silence: bool = True,
        normalize_gain: bool = True,
    ) -> np.ndarray:
        if audio is None or len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure 1D mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        # 1. DC Offset Removal
        audio = audio - float(np.mean(audio))

        # Check for near-total silence
        max_abs = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        if max_abs < 1e-5:
            min_samples = int(sr * cls.MIN_DURATION_SECONDS)
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)), mode="constant")
            return audio

        # 2. Dynamic Energy Silence Trimming
        if trim_silence and len(audio) > int(sr * 0.5):
            frame_len = int(sr * 0.02)  # 20ms
            hop_len = int(sr * 0.01)    # 10ms
            if len(audio) > frame_len:
                energies = [
                    float(np.sqrt(np.mean(audio[i : i + frame_len] ** 2)))
                    for i in range(0, len(audio) - frame_len, hop_len)
                ]
                if energies:
                    max_e = max(energies)
                    threshold = max(0.003, max_e * cls.SILENCE_THRESHOLD_RATIO)
                    active = [idx for idx, e in enumerate(energies) if e > threshold]
                    if active:
                        pad_samples = int(sr * cls.PADDING_DURATION_SECONDS)
                        start = max(0, active[0] * hop_len - pad_samples)
                        end = min(len(audio), (active[-1] + 1) * hop_len + pad_samples)
                        audio = audio[start:end]

        # 3. Peak Normalization
        if normalize_gain:
            peak = float(np.max(np.abs(audio)))
            if peak > 1e-5:
                audio = (audio / peak) * cls.TARGET_PEAK

        # 4. Minimum Duration Safeguard
        min_samples = int(sr * cls.MIN_DURATION_SECONDS)
        if len(audio) < min_samples:
            pad_needed = min_samples - len(audio)
            audio = np.pad(audio, (0, pad_needed), mode="constant")

        return audio


class STTPostProcessor:
    """
    Domain-aware text cleaner and phonetic normalizer for Speech-to-Text outputs:
    1. Removes ASR artifacts (stray ellipses, quotation marks, repetitive punctuation)
    2. Normalizes contractions ('where\'s' -> 'where is', 'what\'s' -> 'what is')
    3. Normalizes phonetic misrecognitions of COCO target classes and keywords
    4. Cleans trailing spaces and capitalization while preserving conversational flow
    """

    PHONETIC_REPLACEMENTS = [
        # Compounds & standard class names
        (r"\bcell\s*phones?\b", "cell phone"),
        (r"\bcellphones?\b", "cell phone"),
        (r"\bsell\s*phones?\b", "cell phone"),
        (r"\bsmart\s*phones?\b", "cell phone"),
        (r"\blap\s*atops?\b", "laptop"),
        (r"\blab\s*tops?\b", "laptop"),
        (r"\blabtop\b", "laptop"),
        (r"\btee\s*vee\b", "tv"),
        (r"\bt\.v\.\b", "tv"),
        (r"\btelevision\s*set\b", "tv"),
        (r"\bwine\s*glasses\b", "wine glass"),
        (r"\bwineglass\b", "wine glass"),
        (r"\bdining\s*tables?\b", "dining table"),
        (r"\bdiningtable\b", "dining table"),
        (r"\bpotted\s*plants?\b", "potted plant"),
        (r"\bpottedplant\b", "potted plant"),
        (r"\btraffic\s*lights?\b", "traffic light"),
        (r"\btrafficlight\b", "traffic light"),
        (r"\bfire\s*hydrants?\b", "fire hydrant"),
        (r"\bfirehydrant\b", "fire hydrant"),
        (r"\bstop\s*signs?\b", "stop sign"),
        (r"\bstopsign\b", "stop sign"),
        (r"\bparking\s*meters?\b", "parking meter"),
        (r"\bparkingmeter\b", "parking meter"),
        (r"\bsports\s*balls?\b", "sports ball"),
        (r"\bsportsball\b", "sports ball"),
        (r"\bbaseball\s*bats?\b", "baseball bat"),
        (r"\bbaseballbat\b", "baseball bat"),
        (r"\bbaseball\s*gloves?\b", "baseball glove"),
        (r"\bbaseballglove\b", "baseball glove"),
        (r"\btennis\s*rackets?\b", "tennis racket"),
        (r"\btennisracket\b", "tennis racket"),
        (r"\bhot\s*dogs?\b", "hot dog"),
        (r"\bhotdog\b", "hot dog"),
        (r"\bteddy\s*bears?\b", "teddy bear"),
        (r"\bstudy\s*bear\b", "teddy bear"),
        (r"\bhair\s*dryers?\b", "hair drier"),
        (r"\bhair\s*driers?\b", "hair drier"),
        (r"\bhairdrier\b", "hair drier"),
        (r"\bhairdryer\b", "hair drier"),
        # Common phonetic speech confusions
        (r"\bsea\s+(?:the|a|my)\s+bottle\b", "see the bottle"),
        (r"\bsea\s+bottle\b", "see bottle"),
        (r"\bbottl\b", "bottle"),
        (r"\bcheir\b", "chair"),
        (r"\bcheer\b(?=\s*(?:on|in|near|beside|to the))", "chair"),
        (r"\bcub\b(?=\s*(?:on|in|near|beside|to the))", "cup"),
    ]

    CONTRACTIONS = [
        (r"\bwhere\'s\b", "where is"),
        (r"\bwhat\'s\b", "what is"),
        (r"\bwho\'s\b", "who is"),
        (r"\bthere\'s\b", "there is"),
        (r"\bcan\'t\b", "cannot"),
        (r"\bdon\'t\b", "do not"),
        (r"\bit\'s\b", "it is"),
    ]

    @classmethod
    def process_transcript(cls, raw_text: str) -> str:
        if not raw_text:
            return ""

        text = raw_text.strip()

        # Remove leading/trailing quotes, ellipses, or stray dots
        text = re.sub(r"^[\"\']+|[\"\']+$", "", text).strip()
        text = re.sub(r"^\.{2,}\s*", "", text).strip()
        text = re.sub(r"\s*\.{2,}$", "", text).strip()

        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Expand common contractions
        for pattern, replacement in cls.CONTRACTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Apply phonetic and compound object normalization
        for pattern, replacement in cls.PHONETIC_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()


class BaseSTTService(ABC):
    """Abstract Base Class for all Speech-To-Text providers."""

    @abstractmethod
    def transcribe(self, audio_data: bytes | Path | str, **kwargs) -> str:
        """
        Transcribe audio input to a text sentence.

        Args:
            audio_data: Raw audio bytes, path to audio file (.wav, .mp3, etc.), or raw buffer.
        Returns:
            Transcribed sentence string.
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the STT provider."""
        pass

    @property
    def is_available(self) -> bool:
        """Whether this provider's required dependencies/models are installed."""
        return True


class MockSTTService(BaseSTTService):
    """
    Mock / Development STT Provider.
    Allows testing sentences directly, using pre-recorded samples,
    or echoing user-supplied sentence queries without requiring audio hardware.
    """

    def __init__(self, default_sentence: str = "Where is my cell phone and the water bottle?"):
        self._default_sentence = default_sentence
        self._sample_transcripts = [
            "Where is my cell phone and the water bottle?",
            "Can you detect the red apple on the dining table?",
            "Look at that cute puppy and kitten sitting on the couch",
            "Count how many cars and bicycles are parked outside",
            "Find the backpack and the tennis racket",
            "I see three people standing near the bus",
            "Is there a hot dog or a pizza on the counter?",
            "Find the laptop but do not detect the television",
        ]
        self._sample_idx = 0

    @property
    def provider_name(self) -> str:
        return "mock_stt"

    def set_sentence(self, sentence: str):
        """Set an explicit sentence to be returned by next transcribe call."""
        self._default_sentence = sentence

    def transcribe(self, audio_data: bytes | Path | str, **kwargs) -> str:
        """Return the configured sentence, or cycle through realistic speech samples."""
        if isinstance(audio_data, str) and audio_data.strip():
            # If a text string was provided directly as input, return it
            return audio_data.strip()

        # If explicit default is set and not empty, use it
        if self._default_sentence:
            return self._default_sentence

        # Otherwise cycle through realistic sample transcripts
        sentence = self._sample_transcripts[self._sample_idx % len(self._sample_transcripts)]
        self._sample_idx += 1
        return sentence


class WhisperSTTService(BaseSTTService):
    """
    Local Open-Source Speech-to-Text Service using OpenAI Whisper.
    Runs 100% locally and offline on macOS (Apple Silicon CPU/MPS).
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        download_root: Path | str | None = None,
    ):
        self.model_size = model_size or getattr(settings, "stt_model_name", "base")
        self.device = device or getattr(settings, "stt_device", "cpu")
        self.download_root = Path(
            download_root or getattr(settings, "whisper_models_dir", settings.models_dir / "whisper")
        )
        self._model = None
        self._lock = threading.RLock()
        self._initialized = False

    @property
    def provider_name(self) -> str:
        return f"whisper_{self.model_size}"

    @property
    def is_available(self) -> bool:
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._initialized and self._model is not None:
            return self._model

        with self._lock:
            if self._initialized and self._model is not None:
                return self._model

            try:
                import whisper
                self.download_root.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Loading local Whisper '%s' model on %s (download_root: %s)...",
                    self.model_size,
                    self.device,
                    self.download_root,
                )
                self._model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    download_root=str(self.download_root),
                )
                self._initialized = True
                logger.info("Local Whisper model '%s' loaded successfully.", self.model_size)
                return self._model
            except ImportError:
                raise RuntimeError(
                    "Whisper STT requested but 'openai-whisper' package is not installed. "
                    "Install it with: pip install openai-whisper"
                )
            except Exception as exc:
                logger.error("Failed to load local Whisper model: %s", exc)
                raise RuntimeError(f"Failed to load Whisper model '{self.model_size}': {exc}")

    def switch_model(self, model_size: str):
        """Switch the loaded Whisper model size (e.g. 'base', 'base.en', 'small')."""
        if model_size == self.model_size and self._model is not None:
            return self._model
        with self._lock:
            self.model_size = model_size
            self._model = None
            self._initialized = False
            return self._load_model()

    @staticmethod
    def _detect_audio_suffix(data: bytes) -> str:
        """Detect container format from initial magic bytes for FFmpeg parsing."""
        if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":
            return ".webm"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
            return ".wav"
        if len(data) >= 4 and data[:4] == b"OggS":
            return ".ogg"
        if len(data) >= 3 and (data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)):
            return ".mp3"
        if len(data) >= 4 and data[:4] == b"fLaC":
            return ".flac"
        if len(data) >= 8 and (b"ftyp" in data[:16]):
            return ".m4a"
        return ".webm"  # Default browser MediaRecorder container

    def transcribe(self, audio_data: bytes | Path | str, **kwargs) -> str:
        """Transcribe speech audio and return the text sentence."""
        res = self.transcribe_detailed(audio_data, **kwargs)
        return res["text"]

    def transcribe_detailed(self, audio_data: bytes | Path | str, **kwargs) -> dict[str, Any]:
        """
        Transcribe speech audio with detailed metadata (latency, language, device),
        applying noise/gain preprocessing, beam search decoding, and domain post-processing.
        """
        model = self._load_model()
        kwargs.setdefault("fp16", False)
        kwargs.setdefault("language", getattr(settings, "whisper_language", "en"))
        kwargs.setdefault("initial_prompt", getattr(settings, "whisper_initial_prompt", ""))
        kwargs.setdefault("temperature", getattr(settings, "whisper_temperature", 0.0))
        kwargs.setdefault("beam_size", getattr(settings, "whisper_beam_size", 5))
        kwargs.setdefault("best_of", getattr(settings, "whisper_best_of", 5))
        kwargs.setdefault("patience", getattr(settings, "whisper_patience", 1.0))
        kwargs.setdefault("no_speech_threshold", getattr(settings, "whisper_no_speech_threshold", 0.4))
        kwargs.setdefault("logprob_threshold", getattr(settings, "whisper_logprob_threshold", -0.8))
        kwargs.setdefault("compression_ratio_threshold", getattr(settings, "whisper_compression_ratio_threshold", 2.4))
        kwargs.setdefault("condition_on_previous_text", getattr(settings, "whisper_condition_on_previous_text", False))

        start_time = time.perf_counter()

        # Case 1: Audio file path
        if isinstance(audio_data, (str, Path)):
            path = Path(audio_data)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

            import whisper.audio as wa
            waveform = wa.load_audio(str(path))
            if getattr(settings, "whisper_preprocess_audio", True):
                waveform = AudioPreprocessor.preprocess_waveform(
                    waveform,
                    trim_silence=getattr(settings, "whisper_trim_silence", True),
                    normalize_gain=getattr(settings, "whisper_normalize_gain", True),
                )
            result = model.transcribe(waveform, **kwargs)
            raw_text = result.get("text", "").strip()
            clean_text = STTPostProcessor.process_transcript(raw_text)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "text": clean_text,
                "raw_text": raw_text,
                "language": result.get("language"),
                "duration": result.get("duration"),
                "segments": result.get("segments", []),
                "processing_time_ms": elapsed_ms,
                "model": self.provider_name,
                "model_size": self.model_size,
                "device": self.device,
            }

        # Case 2: In-memory audio bytes (e.g. from browser mic upload)
        if isinstance(audio_data, bytes):
            if len(audio_data) == 0:
                return {
                    "text": "",
                    "raw_text": "",
                    "language": None,
                    "duration": 0.0,
                    "segments": [],
                    "processing_time_ms": 0.0,
                    "model": self.provider_name,
                    "model_size": self.model_size,
                    "device": self.device,
                }

            suffix = self._detect_audio_suffix(audio_data)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            try:
                with os.fdopen(tmp_fd, "wb") as f:
                    f.write(audio_data)

                import whisper.audio as wa
                waveform = wa.load_audio(tmp_path)
                if getattr(settings, "whisper_preprocess_audio", True):
                    waveform = AudioPreprocessor.preprocess_waveform(
                        waveform,
                        trim_silence=getattr(settings, "whisper_trim_silence", True),
                        normalize_gain=getattr(settings, "whisper_normalize_gain", True),
                    )
                result = model.transcribe(waveform, **kwargs)
                raw_text = result.get("text", "").strip()
                clean_text = STTPostProcessor.process_transcript(raw_text)
                elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                return {
                    "text": clean_text,
                    "raw_text": raw_text,
                    "language": result.get("language"),
                    "duration": result.get("duration"),
                    "segments": result.get("segments", []),
                    "processing_time_ms": elapsed_ms,
                    "model": self.provider_name,
                    "model_size": self.model_size,
                    "device": self.device,
                }
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except OSError:
                    pass

        raise ValueError(f"Invalid audio_data format for Whisper transcription: {type(audio_data)}")


class STTManager:
    """
    Speech-To-Text Manager and NLP Pipeline Coordinator.
    Manages STT providers and orchestrates transcription -> NLP object extraction.
    """

    def __init__(self, nlp_extractor: NLPExtractor | None = None):
        self.nlp = nlp_extractor or get_nlp_extractor()
        whisper_service = WhisperSTTService()
        self._providers: dict[str, BaseSTTService] = {
            "whisper": whisper_service,
            "mock": MockSTTService(),
        }

        # Set Whisper as active default if installed and ready
        if whisper_service.is_available:
            self._active_provider_name: str = "whisper"
        else:
            self._active_provider_name: str = "mock"

    @property
    def active_provider_name(self) -> str:
        return self._active_provider_name

    @property
    def active_provider(self) -> BaseSTTService:
        return self._providers[self._active_provider_name]

    def register_provider(self, name: str, provider: BaseSTTService, set_active: bool = False):
        """Register a user-supplied custom STT provider."""
        self._providers[name] = provider
        if set_active:
            self._active_provider_name = name
        logger.info("Registered STT provider '%s' (active=%s)", name, set_active)

    def set_active_provider(self, name: str):
        """Switch the active STT provider."""
        if name not in self._providers:
            raise KeyError(
                f"STT provider '{name}' not found. Available: {list(self._providers.keys())}"
            )
        self._active_provider_name = name

    def set_model_size(self, model_size: str):
        """Switch model size (e.g. 'base', 'base.en', 'small') on the active provider."""
        provider = self.active_provider
        if hasattr(provider, "switch_model"):
            provider.switch_model(model_size)
            logger.info("Switched STT provider model size to '%s'", model_size)

    def get_available_providers(self) -> dict[str, bool]:
        """Return dict of provider name -> availability."""
        return {name: p.is_available for name, p in self._providers.items()}

    def get_status(self) -> dict[str, Any]:
        """Return status information about the active STT engine."""
        provider = self.active_provider
        model_name = getattr(provider, "model_size", "mock")
        model_path = str(getattr(provider, "download_root", "builtin"))
        device = getattr(provider, "device", "cpu")
        return {
            "active_provider": self._active_provider_name,
            "available_providers": self.get_available_providers(),
            "model_name": model_name,
            "model_path": model_path,
            "device": device,
            "is_local": True,
        }

    def transcribe(self, audio_data: bytes | Path | str, provider_name: str | None = None) -> str:
        """Transcribe audio using the active or specified STT provider."""
        provider = self._providers.get(provider_name) if provider_name else self.active_provider
        if not provider:
            raise ValueError(f"STT provider '{provider_name}' is not registered.")
        return provider.transcribe(audio_data)

    def transcribe_detailed(
        self,
        audio_data: bytes | Path | str,
        provider_name: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Transcribe audio and return full timing and metadata."""
        provider = self._providers.get(provider_name) if provider_name else self.active_provider
        if not provider:
            raise ValueError(f"STT provider '{provider_name}' is not registered.")
        if hasattr(provider, "transcribe_detailed"):
            return provider.transcribe_detailed(audio_data, **kwargs)

        t0 = time.perf_counter()
        text = provider.transcribe(audio_data, **kwargs)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "text": text,
            "language": None,
            "duration": None,
            "processing_time_ms": elapsed_ms,
            "model": provider.provider_name,
            "model_size": "custom",
            "device": "cpu",
        }

    def transcribe_with_hint(
        self,
        audio_data: bytes | Path | str,
        hint_text: str | None = None,
        provider_name: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Transcribes audio with dual-channel consensus:
        Fuses local Whisper transcription with client-side speech recognition hint.
        """
        detailed = self.transcribe_detailed(audio_data, provider_name=provider_name, **kwargs)
        transcribed = detailed.get("text", "")

        if not hint_text or not hint_text.strip():
            detailed["fusion_mode"] = "whisper_solo"
            return detailed

        hint_clean = STTPostProcessor.process_transcript(hint_text.strip())
        if not transcribed:
            detailed["text"] = hint_clean
            detailed["fusion_mode"] = "hint_fallback"
            return detailed

        # Dual-channel validation using NLP extraction
        nlp_whisper = self.nlp.extract_target_and_intent(transcribed)
        nlp_hint = self.nlp.extract_target_and_intent(hint_clean)

        if not nlp_whisper.get("valid") and nlp_hint.get("valid"):
            detailed["text"] = hint_clean
            detailed["fusion_mode"] = "hint_recovery"
        elif nlp_whisper.get("valid") and nlp_hint.get("valid"):
            # If hint contains a requested region that Whisper missed, enrich it
            if nlp_hint.get("requested_region") and not nlp_whisper.get("requested_region"):
                detailed["text"] = hint_clean
                detailed["fusion_mode"] = "hint_enriched_region"
            else:
                detailed["fusion_mode"] = "dual_channel_consensus"
        else:
            detailed["fusion_mode"] = "whisper_primary"

        return detailed

    def transcribe_and_extract(
        self,
        audio_data: bytes | Path | str,
        provider_name: str | None = None,
        include_attributes: bool = True,
        allow_fuzzy: bool = True,
        filter_negated: bool = True,
    ) -> tuple[str, str, NLPExtractionResponse]:
        """
        End-to-End Pipeline:
        1. Transcribe audio to sentence text using local STT provider
        2. Extract COCO objects, attributes, and intent using NLP
        """
        provider = self._providers.get(provider_name) if provider_name else self.active_provider
        sentence = provider.transcribe(audio_data)
        nlp_res = self.nlp.extract(
            sentence=sentence,
            include_attributes=include_attributes,
            allow_fuzzy=allow_fuzzy,
            filter_negated=filter_negated,
        )
        return sentence, provider.provider_name, nlp_res


_DEFAULT_STT_MANAGER: STTManager | None = None


def get_stt_manager() -> STTManager:
    """Singleton getter for STTManager."""
    global _DEFAULT_STT_MANAGER
    if _DEFAULT_STT_MANAGER is None:
        _DEFAULT_STT_MANAGER = STTManager()
    return _DEFAULT_STT_MANAGER
