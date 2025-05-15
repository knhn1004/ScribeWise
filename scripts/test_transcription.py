"""
Test script for the transcription service
"""

import asyncio
import os
from services.transcription_service import TranscriptionService
from dotenv import load_dotenv

load_dotenv()


async def test_transcription(youtube_url: str):
    """Test transcription of a YouTube video"""
    print(f"Testing transcription for: {youtube_url}")

    service = TranscriptionService(output_dir="downloads")

    try:
        result = await service.process_video(url=youtube_url)

        print("\nTranscription Results:")
        print(f"Source: {result['source']}")
        print(f"Model: {result['model']}")
        print(f"Segment count: {len(result['segments'])}")

        if len(result["segments"]) > 0:
            print("\nFirst 3 segments:")
            for i, segment in enumerate(result["segments"][:3]):
                print(
                    f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}"
                )

        if "text" in result:
            print(f"\nTotal text length: {len(result['text'])} characters")
            print(f"Text preview: {result['text'][:200]}...")

        print("\nTranscription test completed successfully!")
        return True

    except Exception as e:
        print(f"Error during transcription test: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

    result = asyncio.run(test_transcription(url))

    sys.exit(0 if result else 1)
