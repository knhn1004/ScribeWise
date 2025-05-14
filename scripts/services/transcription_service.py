"""
Transcription Service for ScribeWise
- Retrieves available captions from videos
- Transcribes audio using Groq Whisper API when captions aren't available
- Splits audio into chunks for processing
"""

import os
import tempfile
import json
from typing import Dict, List, Optional, Literal
import yt_dlp
import groq
from pydantic import BaseModel
import asyncio
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Available Groq Whisper models
WHISPER_MODELS = {
    "whisper-large-v3": {
        "name": "Whisper Large V3",
        "description": "State-of-the-art performance with high accuracy for multilingual transcription and translation tasks",
        "languages": "Multilingual",
        "cost_per_hour": "$0.111",
        "word_error_rate": "10.3%",
        "supports_translation": True,
    },
    "whisper-large-v3-turbo": {
        "name": "Whisper Large V3 Turbo",
        "description": "A fine-tuned version of a pruned Whisper Large V3 designed for fast, multilingual transcription tasks",
        "languages": "Multilingual",
        "cost_per_hour": "$0.04",
        "word_error_rate": "12%",
        "supports_translation": False,
    },
    "distil-whisper-large-v3-en": {
        "name": "Distil-Whisper English",
        "description": "A distilled version of OpenAI's Whisper model, designed for faster, lower cost English speech recognition",
        "languages": "English-only",
        "cost_per_hour": "$0.02",
        "word_error_rate": "13%",
        "supports_translation": False,
    },
}

# Default model to use
DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"


class TranscriptionSegment(BaseModel):
    """Model for a segment of transcription"""

    start: float
    end: float
    text: str


class TranscriptionService:
    def __init__(
        self, output_dir: str = "downloads", model: str = DEFAULT_WHISPER_MODEL
    ):
        """Initialize the Transcription Service"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set the model to use
        if model not in WHISPER_MODELS:
            print(
                f"Warning: Model {model} not found in available Groq Whisper models. Using default: {DEFAULT_WHISPER_MODEL}"
            )
            self.model = DEFAULT_WHISPER_MODEL
        else:
            self.model = model

        print(
            f"Using Groq Whisper model: {self.model} - {WHISPER_MODELS[self.model]['name']}"
        )
        print(f"Description: {WHISPER_MODELS[self.model]['description']}")

        # Initialize Groq client if API key is available
        if GROQ_API_KEY:
            self.groq_client = groq.Client(api_key=GROQ_API_KEY)
            print(f"Successfully initialized Groq client for speech-to-text")
        else:
            print(
                "ERROR: GROQ_API_KEY not found in environment variables. Transcription will not work."
            )
            self.groq_client = None

    async def extract_subtitles(self, url: str) -> Optional[List[TranscriptionSegment]]:
        """Extract subtitles from a YouTube video if available"""
        ydl_opts = {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "skip_download": True,
            "quiet": True,
            "outtmpl": f"{self.output_dir}/%(id)s.%(ext)s",
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_id = info["id"]

                # Check if subtitles are available
                if "subtitles" in info and "en" in info["subtitles"]:
                    # Download subtitles
                    ydl_opts = {
                        "writesubtitles": True,
                        "subtitleslangs": ["en"],
                        "skip_download": True,
                        "quiet": True,
                        "outtmpl": f"{self.output_dir}/{video_id}",
                    }

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                        ydl2.extract_info(url, download=True)

                    # Find and read the subtitle file
                    subtitle_files = [
                        f
                        for f in os.listdir(self.output_dir)
                        if f.startswith(video_id) and f.endswith(".vtt")
                    ]

                    if subtitle_files:
                        # Parse the VTT file
                        segments = self._parse_vtt(
                            f"{self.output_dir}/{subtitle_files[0]}"
                        )
                        return segments

            # If we get here, no subtitles were found
            return None

        except Exception as e:
            print(f"Error extracting subtitles: {str(e)}")
            return None

    def _parse_vtt(self, vtt_file: str) -> List[TranscriptionSegment]:
        """Parse a VTT subtitle file into segments"""
        segments = []

        with open(vtt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for timestamp lines
            if "-->" in line:
                # Extract timestamps
                times = line.split("-->")
                start_str = times[0].strip()
                end_str = times[1].strip().split(" ")[0]  # Remove styling info

                # Convert to seconds
                start = self._timestamp_to_seconds(start_str)
                end = self._timestamp_to_seconds(end_str)

                # Get text (may span multiple lines)
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and not ("-->" in lines[i]):
                    text_lines.append(lines[i].strip())
                    i += 1

                text = " ".join(text_lines)

                # Add segment
                if text:
                    segments.append(
                        TranscriptionSegment(start=start, end=end, text=text)
                    )
            else:
                i += 1

        return segments

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert a timestamp (HH:MM:SS.mmm) to seconds"""
        parts = timestamp.replace(",", ".").split(":")

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        else:
            return float(parts[0])

    async def download_audio(self, url: str) -> str:
        """Download just the audio from a YouTube video"""
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": f"{self.output_dir}/%(id)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = f"{self.output_dir}/{info['id']}.mp3"
                return audio_path
        except Exception as e:
            raise Exception(f"Error downloading audio: {str(e)}")

    async def preprocess_audio(self, audio_path: str) -> str:
        """Optimize audio for transcription using ffmpeg (16kHz mono)"""
        # Create a preprocessed version to optimize for speech recognition
        base_name = os.path.splitext(audio_path)[0]
        processed_path = f"{base_name}_processed.wav"

        # Use ffmpeg to convert to 16kHz mono WAV (optimal for Whisper)
        cmd = f"ffmpeg -i {audio_path} -ar 16000 -ac 1 -c:a pcm_s16le {processed_path} -y -loglevel quiet"
        os.system(cmd)

        if os.path.exists(processed_path):
            print(f"Preprocessed audio saved to {processed_path}")
            return processed_path
        else:
            print(f"Audio preprocessing failed, using original file")
            return audio_path

    async def split_audio(self, audio_path: str, chunk_size_mb: int = 10) -> List[str]:
        """Split audio into chunks of specified size in MB"""
        # Use ffmpeg to split file by size
        base_name = os.path.splitext(audio_path)[0]
        chunk_dir = f"{base_name}_chunks"
        os.makedirs(chunk_dir, exist_ok=True)

        # Get file size in bytes
        file_size = os.path.getsize(audio_path)

        # Calculate number of chunks
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        num_chunks = max(
            1,
            file_size // chunk_size_bytes + (1 if file_size % chunk_size_bytes else 0),
        )

        # Get audio duration
        duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_path}"
        duration_result = os.popen(duration_cmd).read().strip()
        duration = float(duration_result)

        # Calculate chunk duration
        chunk_duration = duration / num_chunks

        # Split audio
        chunk_files = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_file = f"{chunk_dir}/chunk_{i}.wav"  # Using WAV for best compatibility with Whisper

            if i < num_chunks - 1:
                # Not the last chunk
                cmd = f"ffmpeg -i {audio_path} -ss {start_time} -t {chunk_duration} -ar 16000 -ac 1 -c:a pcm_s16le {chunk_file} -y -loglevel quiet"
            else:
                # Last chunk (to handle any rounding issues)
                cmd = f"ffmpeg -i {audio_path} -ss {start_time} -ar 16000 -ac 1 -c:a pcm_s16le {chunk_file} -y -loglevel quiet"

            os.system(cmd)
            chunk_files.append(chunk_file)

        return chunk_files

    async def transcribe_audio_chunk(
        self,
        audio_path: str,
        language: str = None,
        response_format: Literal["json", "verbose_json", "text"] = "verbose_json",
        timestamp_granularities: List[str] = ["segment"],
    ) -> List[TranscriptionSegment]:
        """Transcribe an audio file using Groq API"""
        if not self.groq_client:
            raise Exception("Groq API client not initialized (missing GROQ_API_KEY)")

        # Call Groq Whisper API
        try:
            # Open the file in binary mode to get file object or bytes
            with open(audio_path, "rb") as audio_file:
                request_params = {
                    "file": audio_file,  # Pass the file object directly
                    "model": self.model,
                    "response_format": response_format,
                    "temperature": 0,  # Recommended setting for transcription
                }

                # Add optional parameters if provided
                if language:
                    request_params["language"] = language

                if response_format == "verbose_json" and timestamp_granularities:
                    request_params["timestamp_granularities"] = timestamp_granularities

                # Make the API call
                print(
                    f"Transcribing with Groq: {os.path.basename(audio_path)} using model {self.model}"
                )
                response = self.groq_client.audio.transcriptions.create(
                    **request_params
                )

            # Assume response contains segments
            segments = []

            # Handle different response formats
            if response_format == "text":
                # If response is just text, create a single segment
                segments.append(
                    TranscriptionSegment(
                        start=0.0,
                        end=0.0,  # We don't know the end time
                        text=response.text,
                    )
                )
            elif response_format in ["json", "verbose_json"]:
                # Extract segments from JSON response
                if hasattr(response, "segments"):
                    # Response is an object with segments attribute
                    response_segments = response.segments
                elif isinstance(response, dict) and "segments" in response:
                    # Response is a dictionary with segments key
                    response_segments = response["segments"]
                else:
                    # No segments found, use full text
                    text = (
                        response.text
                        if hasattr(response, "text")
                        else response.get("text", "")
                    )
                    segments.append(TranscriptionSegment(start=0.0, end=0.0, text=text))
                    return segments

                # Process segments
                for seg in response_segments:
                    # Handle both object.attribute and dict["key"] access
                    if isinstance(seg, dict):
                        # Dictionary access
                        start = seg.get("start", 0.0)
                        end = seg.get("end", 0.0)
                        text = seg.get("text", "")
                    else:
                        # Object attribute access
                        start = getattr(seg, "start", 0.0)
                        end = getattr(seg, "end", 0.0)
                        text = getattr(seg, "text", "")

                    segments.append(
                        TranscriptionSegment(start=start, end=end, text=text)
                    )

            return segments

        except Exception as e:
            raise Exception(f"Error transcribing audio with Groq Whisper: {str(e)}")

    async def transcribe_audio(
        self, audio_path: str, language: str = None
    ) -> List[TranscriptionSegment]:
        """Split audio and transcribe all chunks"""
        # First preprocess the audio to optimize for transcription
        processed_audio = await self.preprocess_audio(audio_path)

        # Split audio into chunks
        chunk_files = await self.split_audio(processed_audio)

        # Transcribe each chunk
        all_segments = []
        time_offset = 0.0

        for chunk_file in chunk_files:
            try:
                # Transcribe chunk
                segments = await self.transcribe_audio_chunk(
                    audio_path=chunk_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=[
                        "segment",
                        "word",
                    ],  # Get both segment and word-level timestamps
                )

                # Adjust timestamps and add to result
                for segment in segments:
                    segment.start += time_offset
                    segment.end += time_offset
                    all_segments.append(segment)

                # Update time offset for next chunk
                # This is approximate - in a real implementation, you'd need to get the exact duration
                if segments and segments[-1].end > 0:
                    time_offset = segments[-1].end
                else:
                    # Estimate chunk duration using ffmpeg
                    duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {chunk_file}"
                    duration_result = os.popen(duration_cmd).read().strip()
                    chunk_duration = float(duration_result)
                    time_offset += chunk_duration

            except Exception as e:
                print(f"Error processing chunk {chunk_file}: {str(e)}")

        return all_segments

    async def process_video(self, url: str, language: str = None) -> Dict:
        """Complete process of getting or generating transcription"""
        # First, try to extract subtitles
        subtitles = await self.extract_subtitles(url)

        if subtitles:
            # We have subtitles, use them
            print(f"Using available subtitles for video")
            return {
                "source": "subtitles",
                "segments": [s.dict() for s in subtitles],
                "text": " ".join([s.text for s in subtitles]),
                "model": "subtitles",
            }
        else:
            # No subtitles, download audio and transcribe
            print(
                f"No subtitles found, transcribing audio using Groq Whisper: {self.model}"
            )
            audio_path = await self.download_audio(url)
            transcription = await self.transcribe_audio(audio_path, language)

            return {
                "source": "transcription",
                "segments": [s.dict() for s in transcription],
                "text": " ".join([s.text for s in transcription]),
                "model": self.model,
                "model_info": WHISPER_MODELS[self.model],
            }
