"""
End-to-end test script for ScribeWise
Tests the complete workflow from video processing to summarization
"""

import asyncio
import os
import uuid
from services.imaging_service import ImagingService
from services.transcription_service import TranscriptionService
from services.summarization_service import SummarizationService
from utils.file_utils import save_json, load_json
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = "test_outputs"
DOWNLOAD_DIR = "test_downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


async def run_complete_test(youtube_url: str):
    """Run a complete test of the ScribeWise workflow"""
    print(f"Starting complete test with URL: {youtube_url}")

    request_id = str(uuid.uuid4())
    print(f"Test request ID: {request_id}")

    imaging_service = ImagingService(output_dir=DOWNLOAD_DIR)
    transcription_service = TranscriptionService(output_dir=DOWNLOAD_DIR)
    summarization_service = SummarizationService(output_dir=OUTPUT_DIR)

    try:
        print("\n1. Processing video to extract scenes...")
        video_result = await imaging_service.process_video(str(youtube_url))

        print(f"Video ID: {video_result['video_info']['video_id']}")
        print(f"Title: {video_result['video_info']['title']}")
        print(f"Duration: {video_result['video_info']['duration']} seconds")
        print(f"Extracted {len(video_result['key_frames'])} key frames")

        print("\n2. Transcribing video...")
        transcription_result = await transcription_service.process_video(
            url=str(youtube_url)
        )

        print(f"Transcription source: {transcription_result['source']}")
        print(f"Model used: {transcription_result['model']}")
        print(f"Segments: {len(transcription_result['segments'])}")
        print(f"Text length: {len(transcription_result['text'])} characters")

        print("\n3. Generating summaries and other outputs...")
        content_for_summarization = {
            **transcription_result,
            "video_id": video_result["video_info"]["video_id"],
            "title": video_result["video_info"]["title"],
        }

        outputs = await summarization_service.process_content(content_for_summarization)

        print("\nGenerated outputs:")
        for output_type, output_path in outputs.items():
            if output_type != "error":
                file_exists = os.path.exists(output_path)
                file_size = os.path.getsize(output_path) if file_exists else 0
                print(
                    f"- {output_type}: {output_path} ({'✓' if file_exists else '✗'}, {file_size} bytes)"
                )

        result = {
            "video_info": video_result["video_info"],
            "transcription": transcription_result,
            "outputs": outputs,
            "status": "success",
        }

        if "url" in result["video_info"] and hasattr(
            result["video_info"]["url"], "__str__"
        ):
            result["video_info"]["url"] = str(result["video_info"]["url"])

        result_path = (
            f"{OUTPUT_DIR}/{video_result['video_info']['video_id']}_test_result.json"
        )
        save_json(result, result_path)

        print(f"\nComplete test result saved to: {result_path}")
        print("End-to-end test completed successfully!")
        return True

    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

    result = asyncio.run(run_complete_test(url))

    sys.exit(0 if result else 1)
