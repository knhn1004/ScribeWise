"""
ScribeWise Backend Server
- Provides APIs for video processing, transcription, and summarization
"""

import os
import asyncio
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Depends,
    Request,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import logging
from typing import Dict, List, Optional, Any

# Import services
from services.imaging_service import ImagingService
from services.transcription_service import TranscriptionService
from services.summarization_service import SummarizationService
from services.pdf_service import PDFService

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

# Create a directory to store processed PDFs
PROCESSED_PDF_DIR = os.path.join(Config.OUTPUT_DIR, "processed_pdfs")
os.makedirs(PROCESSED_PDF_DIR, exist_ok=True)

# Log the models being used
logger.info(f"Using Groq text model: {Config.LLM_MODEL}")
logger.info(f"Using Groq speech model: {Config.STT_MODEL}")

# Create service instances
imaging_service = ImagingService(output_dir=Config.DOWNLOAD_DIR)
transcription_service = TranscriptionService(
    output_dir=Config.DOWNLOAD_DIR, model=Config.STT_MODEL
)
summarization_service = SummarizationService(output_dir=Config.OUTPUT_DIR)
pdf_service = PDFService(output_dir=Config.OUTPUT_DIR, download_dir=Config.DOWNLOAD_DIR)

# Log if mermaid-py is available
try:
    import mermaid as md

    logger.info("mermaid-py is available - mindmap image generation is enabled")
except ImportError:
    logger.warning("mermaid-py is not available - mindmap image generation is disabled")

# Check if PyMuPDF is available
try:
    import fitz

    logger.info("PyMuPDF is available - PDF processing is enabled")
except ImportError:
    logger.warning("PyMuPDF is not available - PDF processing is disabled")

# Set up static files to serve output files
app.mount("/outputs", StaticFiles(directory=Config.OUTPUT_DIR), name="outputs")
app.mount("/downloads", StaticFiles(directory=Config.DOWNLOAD_DIR), name="downloads")

# Store in-progress processing tasks
processing_tasks: Dict[str, Dict[str, Any]] = {}
# Store PDF processing tasks
pdf_processing_tasks: Dict[str, Dict[str, Any]] = {}


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
    features = Config.check_optional_features()

    # Check for mermaid-py availability
    try:
        import mermaid as md

        features["mindmap_images"] = True
    except ImportError:
        features["mindmap_images"] = False

    if not Config.check_api_keys():
        return JSONResponse(
            status_code=200,
            content={
                "status": "warning",
                "message": "Missing required API keys",
                "config": Config.get_dict(),
                "features": features,
            },
        )

    return {
        "status": "healthy",
        "config": Config.get_dict(),
        "features": features,
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

        summarization_result = await summarization_service.process_content(
            content_for_summarization
        )

        # Step 4: Save the complete processed result
        complete_result = {
            **video_result,
            "transcription": transcription_result,
            "outputs": summarization_result,
            "status": "success",
            "models": {
                "text_model": Config.LLM_MODEL,
                "speech_model": Config.STT_MODEL,
            },
        }

        video_id = video_result["video_info"]["video_id"]
        output_path = os.path.join(PROCESSED_DIR, f"{video_id}.json")
        save_json(complete_result, output_path)

        processing_tasks[request_id]["status"] = "completed"
        processing_tasks[request_id]["video_id"] = video_id
        processing_tasks[request_id]["outputs"] = summarization_result

        logger.info(f"Video processing completed for {video_id}")

    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        processing_tasks[request_id]["status"] = "failed"
        processing_tasks[request_id]["error"] = str(e)


@app.post("/process", response_model=Dict[str, Any])
async def process_video(
    video_request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    _: Dict = Depends(validate_api_keys),
):
    """Process a video URL, extract frames, generate transcript, and create summaries"""
    request_id = os.urandom(8).hex()

    processing_tasks[request_id] = {
        "status": "initializing",
        "request": video_request.dict(),
        "created_at": asyncio.get_event_loop().time(),
    }

    background_tasks.add_task(process_video_task, video_request, request_id)

    return {
        "status": "processing",
        "message": "Video processing started",
        "request_id": request_id,
    }


@app.get("/status/{request_id}")
async def get_processing_status(request_id: str):
    """Get the status of a video processing request"""
    if request_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Request ID not found")

    task = processing_tasks[request_id]
    response = {
        "status": task["status"],
        "created_at": task["created_at"],
    }

    if task["status"] == "completed":
        response["video_id"] = task["video_id"]
    elif task["status"] == "failed":
        response["error"] = task["error"]
    elif task["status"] in ["downloading", "transcribing", "summarizing"]:
        if "video_info" in task:
            response["video_info"] = task["video_info"]

    return response


@app.get("/videos/{video_id}")
async def get_processed_video(video_id: str):
    """Get a processed video by ID"""
    output_path = os.path.join(PROCESSED_DIR, f"{video_id}.json")

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        result = load_json(output_path)
        return result
    except Exception as e:
        logger.error(f"Error loading video data: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error loading video data: {str(e)}"
        )


@app.get("/mindmap-image/{video_id}")
async def get_mindmap_image(video_id: str):
    """Get the mindmap image for a processed video"""
    output_path = os.path.join(PROCESSED_DIR, f"{video_id}.json")

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        result = load_json(output_path)

        if (
            "outputs" in result
            and "mindmap_image_path" in result["outputs"]
            and "mindmap_image_type" in result["outputs"]
        ):
            image_path = result["outputs"]["mindmap_image_path"]
            image_type = result["outputs"]["mindmap_image_type"]

            if os.path.exists(image_path):
                content_type = (
                    "image/svg+xml" if image_type == "svg" else f"image/{image_type}"
                )

                return FileResponse(
                    image_path,
                    media_type=content_type,
                    filename=f"{video_id}_mindmap.{image_type}",
                )
            else:
                raise HTTPException(
                    status_code=404, detail="Mindmap image file not found"
                )
        else:
            raise HTTPException(
                status_code=404, detail="No mindmap image available for this video"
            )
    except Exception as e:
        logger.error(f"Error getting mindmap image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting mindmap image: {str(e)}"
        )


@app.get("/all-videos")
async def get_all_processed_videos():
    """Get a list of all processed videos"""
    try:
        videos = []
        for filename in os.listdir(PROCESSED_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(PROCESSED_DIR, filename)
                try:
                    data = load_json(file_path)
                    if "video_info" in data:
                        videos.append(
                            {
                                "video_id": data["video_info"]["video_id"],
                                "title": data["video_info"].get("title", "Untitled"),
                                "duration": data["video_info"].get("duration", 0),
                                "processed_date": os.path.getmtime(file_path),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error loading video data from {filename}: {str(e)}")
                    continue

        videos.sort(key=lambda x: x["processed_date"], reverse=True)

        return {
            "status": "success",
            "count": len(videos),
            "videos": videos,
        }
    except Exception as e:
        logger.error(f"Error getting all videos: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting all videos: {str(e)}"
        )


@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    """Get a file from the output directory"""
    full_path = os.path.join(Config.OUTPUT_DIR, file_path)
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": f"Internal server error: {str(exc)}"},
    )


async def process_pdf_task(pdf_path: str, pdf_id: str, request_id: str):
    """Background task to process a PDF"""
    try:
        pdf_processing_tasks[request_id]["status"] = "extracting"

        extraction_result = await pdf_service.extract_text(pdf_path, pdf_id)

        pdf_processing_tasks[request_id]["status"] = "summarizing"

        content_for_summarization = {
            "text": extraction_result["text"],
            "pdf_id": pdf_id,
            "title": extraction_result.get("title", f"PDF {pdf_id}"),
        }

        mindmap_result = await summarization_service.process_content_for_mindmap(
            content_for_summarization
        )

        complete_result = {
            **extraction_result,
            "outputs": mindmap_result,
            "status": "success",
            "models": {
                "text_model": Config.LLM_MODEL,
            },
        }

        output_path = os.path.join(PROCESSED_PDF_DIR, f"{pdf_id}.json")
        save_json(complete_result, output_path)

        pdf_processing_tasks[request_id]["status"] = "completed"
        pdf_processing_tasks[request_id]["pdf_id"] = pdf_id
        pdf_processing_tasks[request_id]["outputs"] = mindmap_result

        logger.info(f"PDF processing completed for {pdf_id}")

    except Exception as e:
        logger.error(f"Error during PDF processing: {str(e)}")
        pdf_processing_tasks[request_id]["status"] = "failed"
        pdf_processing_tasks[request_id]["error"] = str(e)


@app.post("/process-pdf", response_model=Dict[str, Any])
async def process_pdf(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    _: Dict = Depends(validate_api_keys),
):
    """Process a PDF file, extract text, and create summaries"""
    try:
        import fitz
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="PDF processing is not available. PyMuPDF is required.",
        )

    request_id = os.urandom(8).hex()
    pdf_id = os.urandom(8).hex()

    pdf_processing_tasks[request_id] = {
        "status": "initializing",
        "filename": pdf_file.filename,
        "created_at": asyncio.get_event_loop().time(),
    }

    try:
        os.makedirs(Config.DOWNLOAD_DIR, exist_ok=True)
        pdf_path = os.path.join(Config.DOWNLOAD_DIR, f"{pdf_id}.pdf")

        contents = await pdf_file.read()
        with open(pdf_path, "wb") as f:
            f.write(contents)

        pdf_processing_tasks[request_id]["status"] = "uploaded"
        pdf_processing_tasks[request_id]["pdf_path"] = pdf_path

        background_tasks.add_task(process_pdf_task, pdf_path, pdf_id, request_id)

        return {
            "status": "processing",
            "message": "PDF processing started",
            "request_id": request_id,
            "pdf_id": pdf_id,
        }

    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        pdf_processing_tasks[request_id]["status"] = "failed"
        pdf_processing_tasks[request_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@app.get("/pdf-status/{request_id}")
async def get_pdf_processing_status(request_id: str):
    """Get the status of a PDF processing request"""
    if request_id not in pdf_processing_tasks:
        raise HTTPException(status_code=404, detail="Request ID not found")

    task = pdf_processing_tasks[request_id]
    response = {
        "status": task["status"],
        "created_at": task["created_at"],
        "filename": task.get("filename", "Unknown"),
    }

    if task["status"] == "completed":
        response["pdf_id"] = task["pdf_id"]
    elif task["status"] == "failed":
        response["error"] = task["error"]

    return response


@app.get("/pdfs/{pdf_id}")
async def get_processed_pdf(pdf_id: str):
    """Get a processed PDF by ID"""
    output_path = os.path.join(PROCESSED_PDF_DIR, f"{pdf_id}.json")

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    try:
        result = load_json(output_path)
        return result
    except Exception as e:
        logger.error(f"Error loading PDF data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading PDF data: {str(e)}")


@app.get("/pdf-mindmap-image/{pdf_id}")
async def get_pdf_mindmap_image(pdf_id: str):
    """Get the mindmap image for a processed PDF"""
    output_path = os.path.join(PROCESSED_PDF_DIR, f"{pdf_id}.json")

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    try:
        result = load_json(output_path)

        if (
            "outputs" in result
            and "mindmap_image_path" in result["outputs"]
            and "mindmap_image_type" in result["outputs"]
        ):
            image_path = result["outputs"]["mindmap_image_path"]
            image_type = result["outputs"]["mindmap_image_type"]

            if os.path.exists(image_path):
                content_type = (
                    "image/svg+xml" if image_type == "svg" else f"image/{image_type}"
                )

                return FileResponse(
                    image_path,
                    media_type=content_type,
                    filename=f"{pdf_id}_mindmap.{image_type}",
                )
            else:
                raise HTTPException(
                    status_code=404, detail="Mindmap image file not found"
                )
        else:
            raise HTTPException(
                status_code=404, detail="No mindmap image available for this PDF"
            )
    except Exception as e:
        logger.error(f"Error getting PDF mindmap image: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting PDF mindmap image: {str(e)}"
        )


@app.get("/all-pdfs")
async def get_all_processed_pdfs():
    """Get a list of all processed PDFs"""
    try:
        pdfs = []
        for filename in os.listdir(PROCESSED_PDF_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(PROCESSED_PDF_DIR, filename)
                try:
                    data = load_json(file_path)
                    pdfs.append(
                        {
                            "pdf_id": filename.replace(".json", ""),
                            "title": data.get("title", "Untitled"),
                            "num_pages": data.get("num_pages", 0),
                            "processed_date": os.path.getmtime(file_path),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error loading PDF data from {filename}: {str(e)}")
                    continue

        pdfs.sort(key=lambda x: x["processed_date"], reverse=True)

        return {
            "status": "success",
            "count": len(pdfs),
            "pdfs": pdfs,
        }
    except Exception as e:
        logger.error(f"Error getting all PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting all PDFs: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
