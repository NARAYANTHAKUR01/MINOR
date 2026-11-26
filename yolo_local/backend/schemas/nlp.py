from typing import Literal
from pydantic import BaseModel, Field


class ExtractSentenceRequest(BaseModel):
    sentence: str = Field(
        ...,
        description="The spoken or transcribed sentence to analyze (from Speech-To-Text or text prompt)",
        examples=["Can you find the red apple on the dining table?", "Where are my cell phone and keys?"],
    )
    include_attributes: bool = Field(
        default=True,
        description="Extract modifiers, counts, and color attributes if present in the sentence",
    )
    allow_fuzzy: bool = Field(
        default=True,
        description="Allow approximate fuzzy matching for STT phonetic/spelling errors",
    )
    filter_negated: bool = Field(
        default=True,
        description="Exclude negated objects from the primary target_classes list",
    )


class ExtractedObject(BaseModel):
    class_id: int = Field(..., description="COCO Class ID (0 to 79)")
    class_name: str = Field(..., description="Standardized COCO class name (e.g., 'cell phone')")
    matched_phrase: str = Field(..., description="The actual word/phrase extracted from sentence")
    match_type: Literal["exact", "alias", "lemma", "fuzzy"] = Field(
        ..., description="How the phrase was matched (exact match, alias/synonym, lemmatized, or fuzzy)"
    )
    confidence: float = Field(..., description="Match confidence score (0.0 to 1.0)")
    negated: bool = Field(
        default=False,
        description="True if sentence indicates this object should NOT be detected/searched",
    )
    count: int | None = Field(
        default=None,
        description="Specified count or quantity (e.g. 'two dogs' -> 2)",
    )
    attributes: list[str] = Field(
        default_factory=list,
        description="Associated adjectives, colors, or spatial descriptors",
    )
    start_char: int = Field(..., description="Start character offset in the cleaned sentence")
    end_char: int = Field(..., description="End character offset in the cleaned sentence")


class NLPExtractionResponse(BaseModel):
    success: bool = True
    original_sentence: str = Field(..., description="The input raw sentence")
    cleaned_sentence: str = Field(..., description="Normalized sentence after preprocessing")
    intent: str = Field(
        default="general",
        description="Detected user intent: 'locate', 'count', 'detect', 'filter', or 'general'",
    )
    target_classes: list[str] = Field(
        default_factory=list,
        description="List of active non-negated target class names for YOLO detection",
    )
    target_class_ids: list[int] = Field(
        default_factory=list,
        description="List of active target COCO class IDs (0-79) for direct YOLO class filtering",
    )
    excluded_classes: list[str] = Field(
        default_factory=list,
        description="List of explicitly negated or excluded class names (e.g., 'not the car')",
    )
    extracted_objects: list[ExtractedObject] = Field(
        default_factory=list,
        description="Detailed list of all extracted object entities with metadata",
    )
    processing_time_ms: float = Field(
        ..., description="NLP extraction latency in milliseconds"
    )


class ClassDetail(BaseModel):
    class_id: int
    class_name: str
    aliases: list[str] = Field(default_factory=list)
    example_prompts: list[str] = Field(default_factory=list)


class SupportedClassesResponse(BaseModel):
    total_classes: int = 80
    classes: list[ClassDetail]


class TranscribeAndExtractResponse(BaseModel):
    success: bool = True
    transcribed_sentence: str
    stt_provider: str
    nlp_result: NLPExtractionResponse


class TranscribeAudioResponse(BaseModel):
    success: bool = True
    transcription: str = Field(..., description="Transcribed sentence from speech audio")
    provider: str = Field(..., description="STT provider used")
    model_size: str = Field("base", description="Whisper model size used")
    processing_time_ms: float = Field(..., description="Inference latency in milliseconds")
    device: str = Field("cpu", description="Device used for STT inference")
    language: str | None = Field(default=None, description="Detected or specified speech language")


class STTStatusResponse(BaseModel):
    active_provider: str
    available_providers: dict[str, bool]
    model_name: str
    model_path: str
    device: str
    is_local: bool = True
