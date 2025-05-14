# ScribeWise Backend

The ScribeWise backend is a FastAPI service that processes videos to generate study materials.

## Architecture

The backend consists of three main services:

1. **Imaging Service**: Downloads videos and analyzes scenes using OpenCV and PyTorch.
2. **Transcription Service**: Extracts subtitles or transcribes audio using Groq Whisper API.
3. **Summarization Service**: Generates study materials using Langchain and LLMs.

## Environment Setup

1. Create a `.env` file based on `.env.example`:
   ```
   cp .env.example .env
   ```

2. Add your API keys:
   - `GROQ_API_KEY`: Required for LLM and audio transcription
   - `OPENAI_API_KEY`: Optional for RAG functionality

## API Endpoints

- `GET /`: Root endpoint (health check)
- `GET /health`: Detailed health check with API key and feature status
- `POST /process`: Process a video URL and generate study materials
- `GET /status/{request_id}`: Check status of a processing request
- `GET /videos/{video_id}`: Get processed video data
- `GET /files/{file_path}`: Retrieve a generated file

## Running the Backend

### Using Docker

```bash
docker compose up backend
```

### Local Development

```bash
cd scripts
pip install -r requirements.txt
uvicorn main:app --reload
```

## Testing

You can test the backend with the provided test script:

```bash
cd scripts
python test.py <youtube_url>
```

## Generated Outputs

The backend generates several types of study materials:

1. **Markdown Notes**: Comprehensive notes from the video content
2. **Anki Flashcards**: JSON formatted for import into Anki
3. **Mindmaps**: Mermaid syntax diagrams to visualize concepts 