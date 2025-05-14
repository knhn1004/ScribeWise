#!/bin/bash

# Configuration
API_URL="http://localhost:8000"
INPUT="$1"  # First argument is file path or YouTube URL
VERBOSE=false

if [ "$2" = "-v" ] || [ "$2" = "--verbose" ]; then
  VERBOSE=true
fi

if [ -z "$INPUT" ]; then
  echo "Usage: $0 <youtube_url_or_file_path> [-v|--verbose]"
  echo "Examples:"
  echo "  $0 https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  echo "  $0 https://www.youtube.com/watch?v=dQw4w9WgXcQ -v"
  exit 1
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Testing ScribeWise API endpoints${NC}"
echo -e "${BLUE}================================${NC}"

# Check if input is YouTube URL or local file
is_youtube_url() {
  [[ "$1" =~ ^https?://(www\.)?(youtube\.com|youtu\.be) ]]
  return $?
}

# Process based on input type
if is_youtube_url "$INPUT"; then
  echo -e "\n${GREEN}Detected YouTube URL${NC}"
  
  # Process YouTube URL
  echo -e "\n${GREEN}Submitting YouTube URL for processing...${NC}"
  if [ "$VERBOSE" = true ]; then
    echo "POST ${API_URL}/process"
    echo "{\"url\":\"$INPUT\"}"
  fi
  
  RESPONSE=$(curl -s -X POST "${API_URL}/process" \
    -H "Content-Type: application/json" \
    -d "{\"url\":\"$INPUT\"}" \
    -w "\n%{http_code}")
else
  echo -e "${RED}Currently only YouTube URLs are supported in the API${NC}"
  echo "Example: $0 https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  exit 1
fi

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$VERBOSE" = true ]; then
  echo -e "${YELLOW}Response:${NC}"
  echo "$RESPONSE_BODY" | jq . || echo "$RESPONSE_BODY"
  echo -e "${YELLOW}HTTP Status:${NC} $HTTP_CODE"
fi

if [ "$HTTP_CODE" -eq 200 ] || [ "$HTTP_CODE" -eq 202 ]; then
  echo -e "${GREEN}Content submitted successfully${NC}"
  # Extract request_id from response
  REQUEST_ID=$(echo "$RESPONSE_BODY" | grep -o '"request_id":"[^"]*' | sed 's/"request_id":"//')
  
  if [ -z "$REQUEST_ID" ]; then
    echo -e "${RED}Failed to extract request_id from response${NC}"
    echo "Response: $RESPONSE_BODY"
    exit 1
  fi
  
  echo "Request ID: $REQUEST_ID"
else
  echo -e "${RED}Failed to submit content${NC}"
  echo "Response: $RESPONSE_BODY"
  echo "HTTP Status: $HTTP_CODE"
  exit 1
fi

# 2. Check processing status
echo -e "\n${GREEN}Checking processing status...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop checking${NC}"

VIDEO_ID=""
while true; do
  STATUS_RESPONSE=$(curl -s -X GET "${API_URL}/status/$REQUEST_ID")
  STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*' | sed 's/"status":"//')
  
  echo "Status: $STATUS"
  
  if [ "$VERBOSE" = true ]; then
    echo -e "${YELLOW}Details:${NC}"
    echo "$STATUS_RESPONSE" | jq . || echo "$STATUS_RESPONSE"
  fi
  
  if [ "$STATUS" = "complete" ]; then
    echo -e "${GREEN}Processing completed!${NC}"
    # Extract video_id for further requests
    VIDEO_ID=$(echo "$STATUS_RESPONSE" | grep -o '"video_id":"[^"]*' | sed 's/"video_id":"//')
    if [ -z "$VIDEO_ID" ]; then
      # Try to extract from result or video_info
      VIDEO_ID=$(echo "$STATUS_RESPONSE" | grep -o '"video_id":"[^"]*' | head -1 | sed 's/"video_id":"//')
    fi
    break
  elif [ "$STATUS" = "error" ]; then
    echo -e "${RED}Processing failed${NC}"
    if [ "$VERBOSE" = true ]; then
      echo -e "${YELLOW}Error details:${NC}"
      echo "$STATUS_RESPONSE" | jq . || echo "$STATUS_RESPONSE"
    else
      echo "Response: $STATUS_RESPONSE"
    fi
    exit 1
  fi
  
  sleep 5
done

if [ -z "$VIDEO_ID" ]; then
  echo -e "${RED}Failed to extract video_id from status response${NC}"
  echo "Will try to use the request ID instead"
  VIDEO_ID=$REQUEST_ID
fi

# 3. Get the processed video details
echo -e "\n${GREEN}Getting processed video details...${NC}"
VIDEO_DETAILS=$(curl -s -X GET "${API_URL}/videos/$VIDEO_ID")
if [ "$VERBOSE" = true ]; then
  echo "$VIDEO_DETAILS" | jq .
else
  echo "$VIDEO_DETAILS" | jq -r '.video_info.title'
fi

# 4. Get outputs if they exist
echo -e "\n${GREEN}Checking for generated outputs...${NC}"

# Function to check if path exists in output
check_output() {
  local path=$1
  local desc=$2
  
  # Try to get the file
  local response=$(curl -s -I -X GET "${API_URL}${path}" -w "%{http_code}")
  local http_code=$(echo "$response" | grep "HTTP" | awk '{print $2}')
  
  if [[ "$http_code" == "200" ]]; then
    echo -e "${GREEN}✓ ${desc} is available at: ${path}${NC}"
    echo -e "  curl -s -X GET \"${API_URL}${path}\" > ${desc// /_}.output"
  else
    echo -e "${RED}✗ ${desc} not available${NC}"
  fi
}

# Check for common output paths
OUTPUT_PATHS=$(curl -s -X GET "${API_URL}/videos/$VIDEO_ID" | jq -r '.outputs | select(.!=null) | to_entries[] | .key + ":" + .value')

if [ -n "$OUTPUT_PATHS" ]; then
  echo -e "${GREEN}Found output paths:${NC}"
  echo "$OUTPUT_PATHS" | while IFS=: read -r key value; do
    echo -e "${GREEN}${key}: ${value}${NC}"
    check_output "$value" "$key"
  done
else
  echo -e "${RED}No specific output paths found in response${NC}"
fi

echo -e "\n${GREEN}Testing completed successfully!${NC}" 