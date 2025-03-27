import os
from groq import Groq
import logging
import sys
from pathlib import Path
import yt_dlp
import re
from dotenv import load_dotenv
import requests
from pydub import AudioSegment
import tempfile

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 39 * 1024 * 1024
CHUNK_DURATION = 10 * 60 * 1000


def is_youtube_url(url):
    youtube_regex = (
        r"(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/"
        r"(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})"
    )
    return bool(re.match(youtube_regex, url))


def download_youtube_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": "%(title)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    try:
        logger.info("Downloading audio from YouTube...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_file = f"{info['title']}.mp3"
            logger.info(f"Downloaded: {downloaded_file}")
            return downloaded_file
    except Exception as e:
        logger.error(f"Error downloading YouTube audio: {str(e)}")
        raise


def configure_groq():
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        logger.error(
            "GROQ_API_KEY environment variable not found. Please set your API key."
        )
        sys.exit(1)

    client = Groq(api_key=GROQ_API_KEY)
    return client


def split_audio_file(file_path):
    file_size = os.path.getsize(file_path)
    if file_size <= MAX_FILE_SIZE:
        return [file_path]

    logger.info(
        f"File size ({file_size} bytes) exceeds limit. Splitting into chunks..."
    )

    audio = AudioSegment.from_file(file_path)
    chunks = []

    total_duration = len(audio)
    num_chunks = (total_duration // CHUNK_DURATION) + (
        1 if total_duration % CHUNK_DURATION else 0
    )

    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for chunks: {temp_dir}")

    for i in range(num_chunks):
        start = i * CHUNK_DURATION
        end = min((i + 1) * CHUNK_DURATION, total_duration)

        chunk = audio[start:end]
        chunk_path = os.path.join(temp_dir, f"chunk_{i+1}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

        logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")

    return chunks


def transcribe_audio(groq_client: Groq, file_path: str, language="en"):
    try:
        chunk_files = split_audio_file(file_path)
        full_transcript = []

        for i, chunk_path in enumerate(chunk_files, 1):
            if len(chunk_files) > 1:
                logger.info(f"Processing chunk {i}/{len(chunk_files)}")

            with open(chunk_path, "rb") as file:
                file_content = file.read()
                logger.info(f"Chunk size: {len(file_content)} bytes")

                url = "https://api.groq.com/openai/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {groq_client.api_key}"}
                files = {
                    "file": (os.path.basename(chunk_path), file_content, "audio/mpeg")
                }
                data = {
                    "model": "whisper-large-v3-turbo",
                    "prompt": "Transcribe the following audio",
                    "response_format": "verbose_json",
                    "language": language,
                    "temperature": 0.0,
                }

                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()

                result = response.json()
                transcription_text = result.get("text", "")
                full_transcript.append(transcription_text)

                if "segments" in result:
                    avg_confidence = sum(
                        seg.get("avg_logprob", 0) for seg in result["segments"]
                    ) / len(result["segments"])
                    logger.info(f"Chunk {i} average confidence score: {avg_confidence}")

        if len(chunk_files) > 1:
            temp_dir = os.path.dirname(chunk_files[0])
            for chunk_path in chunk_files:
                os.remove(chunk_path)
            os.rmdir(temp_dir)
            logger.info("Cleaned up temporary chunk files")

        return " ".join(full_transcript)

    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise


def save_transcript(text, original_file_path):
    output_path = Path(original_file_path).with_suffix(".txt")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Transcript saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving transcript: {str(e)}")
        raise


def main():
    user_input = input("Enter the path to your MP4/MP3 file or YouTube URL: ").strip()

    file_to_transcribe = None
    is_temp_file = False

    try:
        if is_youtube_url(user_input):
            file_to_transcribe = download_youtube_audio(user_input)
            is_temp_file = True
        else:
            if not os.path.exists(user_input):
                logger.error(f"File not found: {user_input}")
                sys.exit(1)

            if not user_input.lower().endswith((".mp4", ".mp3")):
                logger.error("Please provide an MP4 or MP3 file")
                sys.exit(1)

            file_to_transcribe = user_input

        groq_client = configure_groq()

        print("\nStarting transcription...\n")
        transcript = transcribe_audio(groq_client, file_to_transcribe)

        print("\nTranscription Result:")
        print("-" * 50)
        print(transcript)
        print("-" * 50)

        save_transcript(transcript, file_to_transcribe)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        if is_temp_file and file_to_transcribe and os.path.exists(file_to_transcribe):
            try:
                logger.info(f"Cleaned up temporary file: {file_to_transcribe}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")


if __name__ == "__main__":
    main()
