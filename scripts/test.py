"""
Test script to verify the ScribeWise backend functionality
"""

import os
import sys
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the API URL
API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def process_video(url):
    """Test video processing"""
    print(f"Processing video: {url}")
    response = requests.post(
        f"{API_URL}/process",
        json={"url": url}
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    request_id = result.get("request_id")
    print(f"Request ID: {request_id}")
    return request_id


def check_status(request_id):
    """Check the status of a processing request"""
    print(f"Checking status for request: {request_id}")
    response = requests.get(f"{API_URL}/status/{request_id}")
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    return response.json()


def main():
    """Main test function"""
    # Check if the API is running
    if not test_health():
        print("Health check failed. Is the API running?")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test.py <youtube_url>")
        sys.exit(1)
    
    # Get the YouTube URL from command line
    youtube_url = sys.argv[1]
    
    # Process the video
    request_id = process_video(youtube_url)
    if not request_id:
        print("Failed to start processing")
        sys.exit(1)
    
    # Poll for status
    max_attempts = 60  # 5 minutes max (5 seconds between checks)
    attempts = 0
    
    while attempts < max_attempts:
        status_data = check_status(request_id)
        
        if not status_data:
            print("Failed to get status")
            break
        
        status = status_data.get("status")
        print(f"Status: {status}")
        
        if status == "complete":
            print("Processing complete!")
            
            # Print output paths
            if "result" in status_data and "outputs" in status_data["result"]:
                outputs = status_data["result"]["outputs"]
                print("\nGenerated outputs:")
                for output_type, path in outputs.items():
                    if output_type.endswith("_path"):
                        print(f"- {output_type}: {path}")
                        
            break
            
        elif status == "error":
            print(f"Error: {status_data.get('error', 'Unknown error')}")
            break
            
        # Wait before checking again
        time.sleep(5)
        attempts += 1
    
    if attempts >= max_attempts:
        print("Timeout waiting for processing to complete")


if __name__ == "__main__":
    main() 