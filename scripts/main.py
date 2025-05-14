"""
ScribeWise Backend Server
- Provides APIs for video processing, transcription, and summarization
"""

import os
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
from typing import Dict, List, Optional, Any

# Import services
from services.imaging_service import ImagingService
from services.transcription_service import TranscriptionService
from services.summarization_service import SummarizationService

# Import models and utilities
from models.schemas import (
    ProcessVideoRequest,
    ProcessVideoResponse,
    VideoInfo,
    ErrorResponse,
)
from utils.config import Config, GROQ_TEXT_MODELS, GROQ_SPEECH_MODELS
from utils.file_utils import ensure_directory, save_json, load_json

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scribewise")

# Set up the application
app = FastAPI(
    title="ScribeWise API",
    description="Backend API for ScribeWise video transcription and summarization service",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up directories
Config.setup()

# Create a directory to store processed videos
PROCESSED_DIR = os.path.join(Config.OUTPUT_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Log the models being used
logger.info(f"Using Groq text model: {Config.LLM_MODEL}")
logger.info(f"Using Groq speech model: {Config.STT_MODEL}")

# Create service instances
imaging_service = ImagingService(output_dir=Config.DOWNLOAD_DIR)
transcription_service = TranscriptionService(
    output_dir=Config.DOWNLOAD_DIR, model=Config.STT_MODEL
)
summarization_service = SummarizationService(output_dir=Config.OUTPUT_DIR)

# Set up static files to serve output files
app.mount("/outputs", StaticFiles(directory=Config.OUTPUT_DIR), name="outputs")
app.mount("/downloads", StaticFiles(directory=Config.DOWNLOAD_DIR), name="downloads")

# Store in-progress processing tasks
processing_tasks: Dict[str, Dict[str, Any]] = {}


def validate_api_keys():
    """Check if required API keys are present"""
    if not Config.check_api_keys():
        raise HTTPException(
            status_code=500,
            detail="Missing required API keys. Please check your .env file.",
        )


@app.get("/")
def read_root():
    """Root endpoint to check if the API is running"""
    return {"message": "Welcome to the ScribeWise backend!"}


@app.get("/health")
def health_check():
    """Health check endpoint that verifies API keys and system status"""
    if not Config.check_api_keys():
        return JSONResponse(
            status_code=200,
            content={
                "status": "warning",
                "message": "Missing required API keys",
                "config": Config.get_dict(),
                "features": Config.check_optional_features(),
            },
        )

    return {
        "status": "healthy",
        "config": Config.get_dict(),
        "features": Config.check_optional_features(),
    }


@app.get("/models")
def get_available_models():
    """Get all available models supported by the API"""
    return {
        "status": "success",
        "models": Config.get_available_models(),
        "current_config": {
            "text_model": Config.LLM_MODEL,
            "speech_model": Config.STT_MODEL,
        },
    }


async def process_video_task(video_request: ProcessVideoRequest, request_id: str):
    """Background task to process a video"""
    try:
        # Update task status
        processing_tasks[request_id]["status"] = "downloading"

        # Step 1: Download and analyze video scenes
        video_result = await imaging_service.process_video(str(video_request.url))

        # Update task status
        processing_tasks[request_id]["status"] = "transcribing"
        processing_tasks[request_id]["video_info"] = video_result["video_info"]

        # Step 2: Transcribe the video
        # Get language from request if specified, otherwise use None for auto-detection
        language = (
            video_request.language if hasattr(video_request, "language") else None
        )
        transcription_result = await transcription_service.process_video(
            url=str(video_request.url), language=language
        )

        # Update task status
        processing_tasks[request_id]["status"] = "summarizing"

        # Step 3: Generate summaries and other outputs
        # Combine video and transcription info for the summarization
        content_for_summarization = {
            **transcription_result,
            "video_id": video_result["video_info"]["video_id"],
            "title": video_result["video_info"]["title"],
        }

        outputs = await summarization_service.process_content(content_for_summarization)

        # Update task status to complete
        processing_tasks[request_id]["status"] = "complete"
        processing_tasks[request_id]["outputs"] = outputs

        # Create response data
        video_info = VideoInfo(
            video_id=video_result["video_info"]["video_id"],
            title=video_result["video_info"]["title"],
            duration=video_result["video_info"]["duration"],
            url=video_request.url,
        )

        response_data = {
            "video_info": video_info.dict(),
            "scenes": [
                {
                    "scene_idx": i,
                    "start_time": time,
                    "frame_path": frame,
                    "ocr_text": None,  # OCR would be implemented in a more complete version
                }
                for i, (time, frame) in enumerate(
                    zip(video_result["scene_times"], video_result["key_frames"])
                )
            ],
            "transcription": transcription_result,
            "outputs": outputs,
            "status": "success",
            "models": {
                "text_model": Config.LLM_MODEL,
                "speech_model": transcription_result.get("model", Config.STT_MODEL),
            },
        }

        # Save the processed result to disk for later retrieval
        processed_file = os.path.join(
            PROCESSED_DIR, f"{video_result['video_info']['video_id']}.json"
        )
        save_json(response_data, processed_file)

        # Update the task data one last time
        processing_tasks[request_id]["result"] = response_data

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        processing_tasks[request_id]["status"] = "error"
        processing_tasks[request_id]["error"] = str(e)


@app.post("/process", response_model=Dict[str, Any])
async def process_video(
    video_request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    _: Dict = Depends(validate_api_keys),
):
    """
    Process a video:
    1. Download and analyze video scenes
    2. Transcribe the video
    3. Generate summaries, flashcards, and mindmaps
    """
    # Create a unique request ID
    import uuid

    request_id = str(uuid.uuid4())

    # Initialize task entry
    processing_tasks[request_id] = {
        "status": "queued",
        "request": video_request.dict(),
        "created_at": asyncio.get_event_loop().time(),
    }

    # Start processing in the background
    background_tasks.add_task(process_video_task, video_request, request_id)

    # Return the task ID for status checking
    return {
        "request_id": request_id,
        "status": "queued",
        "message": "Video processing started",
        "models": {"text_model": Config.LLM_MODEL, "speech_model": Config.STT_MODEL},
    }


@app.get("/status/{request_id}")
async def get_processing_status(request_id: str):
    """Get the status of a processing request"""
    if request_id not in processing_tasks:
        raise HTTPException(
            status_code=404, detail=f"Request ID {request_id} not found"
        )

    task = processing_tasks[request_id]
    response = {"request_id": request_id, "status": task["status"]}

    # Include more details based on status
    if task["status"] == "complete":
        if "result" in task:
            response["result"] = task["result"]
        elif "outputs" in task:
            response["outputs"] = task["outputs"]
    elif task["status"] == "error":
        response["error"] = task.get("error", "Unknown error")
    elif task["status"] in ["downloading", "transcribing", "summarizing"]:
        response["progress"] = task["status"]
        if "video_info" in task:
            response["video_info"] = task["video_info"]

    return response


@app.get("/videos/{video_id}")
async def get_processed_video(video_id: str):
    """Get a processed video by ID"""
    processed_file = os.path.join(PROCESSED_DIR, f"{video_id}.json")

    if not os.path.exists(processed_file):
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    try:
        data = load_json(processed_file)
        return data
    except Exception as e:
        logger.error(f"Error loading processed video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error loading video data: {str(e)}"
        )


@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    """Get a file from the output directory"""
    full_path = os.path.join(Config.OUTPUT_DIR, file_path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File {file_path} not found")

    return FileResponse(full_path)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with the ErrorResponse model"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error", error=exc.detail, details=exc.headers
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with the ErrorResponse model"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error", error="Internal server error", details={"message": str(exc)}
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    # Start the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
