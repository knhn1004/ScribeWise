"""
Pydantic models for ScribeWise API request and response validation
"""

from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime


class ProcessVideoRequest(BaseModel):
    """Request model for processing a video"""

    url: HttpUrl = Field(..., description="YouTube video URL to process")
    output_format: Optional[List[str]] = Field(
        default=["notes", "flashcards", "mindmap"],
        description="List of output formats to generate",
    )
    skip_download: bool = Field(
        default=False, description="Skip video/audio download if already processed"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code in ISO-639-1 format (e.g., 'en' for English). If not provided, auto-detection will be used.",
    )
    text_model: Optional[str] = Field(
        default=None,
        description="Groq text model to use for summarization. If not provided, the default from configuration will be used.",
    )
    speech_model: Optional[str] = Field(
        default=None,
        description="Groq speech model to use for transcription. If not provided, the default from configuration will be used.",
    )

    @field_validator("url")
    @classmethod
    def url_must_be_supported(cls, v):
        """Validate that the URL is from a supported platform"""
        supported_platforms = ["youtube.com", "youtu.be", "vimeo.com"]
        if not any(platform in str(v) for platform in supported_platforms):
            raise ValueError(
                f"URL must be from a supported platform: {', '.join(supported_platforms)}"
            )
        return v


class VideoInfo(BaseModel):
    """Basic video information"""

    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    duration: int = Field(..., description="Video duration in seconds")
    url: HttpUrl = Field(..., description="Original video URL")

    def dict(self, *args, **kwargs):
        """Convert model to dictionary, handling HttpUrl objects"""
        # Get the original dict
        result = super().dict(*args, **kwargs)
        # Convert HttpUrl to string
        if "url" in result and hasattr(result["url"], "__str__"):
            result["url"] = str(result["url"])
        return result


class SceneInfo(BaseModel):
    """Information about a detected scene"""

    scene_idx: int = Field(..., description="Scene index (0-based)")
    start_time: float = Field(..., description="Start time in seconds")
    frame_path: str = Field(..., description="Path to the keyframe image")
    ocr_text: Optional[str] = Field(
        None, description="OCR extracted text from the frame"
    )


class TranscriptionSegment(BaseModel):
    """A segment of transcription with timing information"""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class ProcessVideoResponse(BaseModel):
    """Response model for a processed video"""

    video_info: VideoInfo = Field(..., description="Basic video information")
    scenes: Optional[List[SceneInfo]] = Field(
        None, description="Detected scenes, if applicable"
    )
    transcription: Optional[Dict[str, Any]] = Field(
        None, description="Transcription information"
    )
    outputs: Dict[str, str] = Field(..., description="Paths to generated output files")
    status: str = Field("success", description="Processing status")
    models: Optional[Dict[str, str]] = Field(
        None, description="Models used for processing"
    )


class FlashcardModel(BaseModel):
    """Model for an Anki flashcard"""

    front: str = Field(
        ..., description="Front side of the flashcard with the question or concept"
    )
    back: str = Field(
        ..., description="Back side of the flashcard with the answer or explanation"
    )


class FlashcardOutput(BaseModel):
    """Model for a collection of flashcards"""

    flashcards: List[FlashcardModel] = Field(
        ..., description="List of generated flashcards"
    )


class PDFInfo(BaseModel):
    """Information about a processed PDF"""

    pdf_id: str = Field(..., description="Unique identifier for the PDF")
    title: Optional[str] = Field(None, description="Title of the PDF")
    filename: str = Field(..., description="Filename of the PDF")
    page_count: Optional[int] = Field(None, description="Number of pages in the PDF")
    file_size: Optional[int] = Field(None, description="Size of the PDF file in bytes")


class PDFProcessingStatus(BaseModel):
    """Status of a PDF processing request"""

    request_id: str = Field(..., description="Request ID for the PDF processing task")
    status: str = Field(..., description="Status of the processing task")
    pdf_info: Optional[PDFInfo] = Field(None, description="Information about the PDF")
    error: Optional[str] = Field(None, description="Error message, if any")
    results: Optional[Dict[str, Any]] = Field(None, description="Processing results")


class PDFProcessingResult(BaseModel):
    """Result of PDF processing"""

    pdf_info: PDFInfo = Field(..., description="Information about the PDF")
    notes: Dict[str, Any] = Field(..., description="Generated notes")
    flashcards: Dict[str, Any] = Field(..., description="Generated flashcards")
    mindmap: Optional[Dict[str, Any]] = Field(None, description="Generated mindmap")


class ErrorResponse(BaseModel):
    """Error response model"""

    status: str = Field(
        "error", description="Status is always 'error' for error responses"
    )
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details if available"
    )
