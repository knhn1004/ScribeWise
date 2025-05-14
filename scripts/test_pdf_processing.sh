#!/bin/bash

# Script to test ScribeWise PDF processing functionality
# Usage: ./test_pdf_processing.sh /path/to/your/document.pdf

# Configuration
API_URL="http://localhost:8000"
MAX_RETRIES=30
RETRY_DELAY=5  # seconds

# Check if a PDF file was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/your/document.pdf"
    exit 1
fi

PDF_PATH="$1"

# Check if file exists
if [ ! -f "$PDF_PATH" ]; then
    echo "Error: PDF file not found: $PDF_PATH"
    exit 1
fi

echo "=== ScribeWise PDF Processing Test ==="
echo "Testing PDF: $PDF_PATH"

# 1. Upload the PDF file
echo -e "\n[1/3] Uploading PDF file..."
UPLOAD_RESPONSE=$(curl -s -X POST "${API_URL}/process-pdf" \
    -H "Content-Type: multipart/form-data" \
    -F "pdf_file=@${PDF_PATH}")

# Check for errors
if [[ "$UPLOAD_RESPONSE" == *"error"* ]]; then
    echo "Error uploading PDF:"
    echo "$UPLOAD_RESPONSE" | grep -o '"error":"[^"]*"' | cut -d':' -f2 | tr -d '"'
    exit 1
fi

# Extract request_id
REQUEST_ID=$(echo "$UPLOAD_RESPONSE" | grep -o '"request_id":"[^"]*"' | cut -d':' -f2 | tr -d '"')
PDF_ID=$(echo "$UPLOAD_RESPONSE" | grep -o '"pdf_id":"[^"]*"' | cut -d':' -f2 | tr -d '"')

if [ -z "$REQUEST_ID" ]; then
    echo "Error: Failed to get request_id from upload response"
    echo "$UPLOAD_RESPONSE"
    exit 1
fi

echo "PDF uploaded successfully!"
echo "Request ID: $REQUEST_ID"
echo "PDF ID: $PDF_ID"

# 2. Poll for processing status
echo -e "\n[2/3] Checking processing status..."
RETRY_COUNT=0
STATUS="queued"

while [ "$STATUS" != "complete" ] && [ "$STATUS" != "error" ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    let RETRY_COUNT=$RETRY_COUNT+1
    
    echo "Checking status (attempt $RETRY_COUNT of $MAX_RETRIES)..."
    STATUS_RESPONSE=$(curl -s "${API_URL}/pdf-status/${REQUEST_ID}")
    
    # Extract status
    STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d':' -f2 | tr -d '"')
    
    if [ "$STATUS" == "error" ]; then
        ERROR_MSG=$(echo "$STATUS_RESPONSE" | grep -o '"error":"[^"]*"' | cut -d':' -f2 | tr -d '"')
        echo "Error processing PDF: $ERROR_MSG"
        exit 1
    elif [ "$STATUS" == "complete" ]; then
        echo "Processing complete!"
        break
    else
        echo "Current status: $STATUS"
        echo "Waiting ${RETRY_DELAY} seconds before checking again..."
        sleep $RETRY_DELAY
    fi
done

if [ "$STATUS" != "complete" ]; then
    echo "Processing timed out after $RETRY_COUNT attempts"
    exit 1
fi

# 3. Get the processing results
echo -e "\n[3/3] Getting processing results..."
RESULTS_RESPONSE=$(curl -s "${API_URL}/pdfs/${PDF_ID}")

# Display summary of results
echo -e "\n=== PDF Processing Results ==="
echo "Title: $(echo "$RESULTS_RESPONSE" | grep -o '"title":"[^"]*"' | head -1 | cut -d':' -f2 | tr -d '"')"
echo "Page count: $(echo "$RESULTS_RESPONSE" | grep -o '"page_count":[0-9]*' | cut -d':' -f2)"

# Check for output files
NOTES_PATH=$(echo "$RESULTS_RESPONSE" | grep -o '"notes_path":"[^"]*"' | cut -d':' -f2 | tr -d '"')
FLASHCARDS_PATH=$(echo "$RESULTS_RESPONSE" | grep -o '"flashcards_path":"[^"]*"' | cut -d':' -f2 | tr -d '"')
MINDMAP_PATH=$(echo "$RESULTS_RESPONSE" | grep -o '"mindmap_path":"[^"]*"' | cut -d':' -f2 | tr -d '"')
MINDMAP_IMAGE_PATH=$(echo "$RESULTS_RESPONSE" | grep -o '"mindmap_image_path":"[^"]*"' | cut -d':' -f2 | tr -d '"')

echo -e "\nGenerated Output Files:"
[ ! -z "$NOTES_PATH" ] && echo "- Notes: $NOTES_PATH"
[ ! -z "$FLASHCARDS_PATH" ] && echo "- Flashcards: $FLASHCARDS_PATH"
[ ! -z "$MINDMAP_PATH" ] && echo "- Mindmap: $MINDMAP_PATH"
[ ! -z "$MINDMAP_IMAGE_PATH" ] && echo "- Mindmap Image: $MINDMAP_IMAGE_PATH"

echo -e "\nTo view the mindmap image, visit: ${API_URL}/pdf-mindmap-image/${PDF_ID}"
echo -e "\nScribeWise PDF processing test completed successfully!" 