"""
Test script for the transcription service
"""

import asyncio
import os
from services.transcription_service import TranscriptionService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_transcription(youtube_url: str):
    """Test transcription of a YouTube video"""
    print(f"Testing transcription for: {youtube_url}")

    # Create transcription service
    service = TranscriptionService(output_dir="downloads")

    # Process video
    try:
        result = await service.process_video(url=youtube_url)

        # Print basic info
        print("\nTranscription Results:")
        print(f"Source: {result['source']}")
        print(f"Model: {result['model']}")
        print(f"Segment count: {len(result['segments'])}")

        # Print first few segments
        if len(result["segments"]) > 0:
            print("\nFirst 3 segments:")
            for i, segment in enumerate(result["segments"][:3]):
                print(
                    f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}"
                )

        # Print total text length
        if "text" in result:
            print(f"\nTotal text length: {len(result['text'])} characters")
            print(f"Text preview: {result['text'][:200]}...")

        print("\nTranscription test completed successfully!")
        return True

    except Exception as e:
        print(f"Error during transcription test: {str(e)}")
        return False


if __name__ == "__main__":
    # Get URL from command line or use a default test URL
    import sys

    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        # Default short test video
        url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" (first YouTube video)

    result = asyncio.run(test_transcription(url))

    # Exit with status code
    sys.exit(0 if result else 1)
