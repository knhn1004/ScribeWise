"""
Configuration utilities for ScribeWise
"""

import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

# Load environment variables from .env file
load_dotenv()

# Groq model configurations
GROQ_TEXT_MODELS = {
    "llama3-70b-8192": {
        "name": "Llama 3 70B",
        "description": "Llama 3 70B with 8K context window",
        "context_window": 8192,
        "max_completion_tokens": 8192,
    },
    "llama3-8b-8192": {
        "name": "Llama 3 8B",
        "description": "Llama 3 8B with 8K context window",
        "context_window": 8192,
        "max_completion_tokens": 8192,
    },
    "gemma2-9b-it": {
        "name": "Gemma 2 9B IT",
        "description": "Gemma 2 with 9B parameters",
        "context_window": 8192,
        "max_completion_tokens": 8192,
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral 8x7B",
        "description": "Mixtral 8x7B with extended 32K context window",
        "context_window": 32768,
        "max_completion_tokens": 8192,
    },
}

# Groq speech-to-text models
GROQ_SPEECH_MODELS = {
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
        "description": "Fine-tuned version of pruned Whisper Large V3 for fast multilingual transcription",
        "languages": "Multilingual",
        "cost_per_hour": "$0.04",
        "word_error_rate": "12%",
        "supports_translation": False,
    },
    "distil-whisper-large-v3-en": {
        "name": "Distil-Whisper English",
        "description": "Distilled version of Whisper model for faster, lower cost English speech recognition",
        "languages": "English-only",
        "cost_per_hour": "$0.02",
        "word_error_rate": "13%",
        "supports_translation": False,
    },
}

# Default models to use
DEFAULT_TEXT_MODEL = "llama3-70b-8192"
DEFAULT_SPEECH_MODEL = "whisper-large-v3-turbo"


class Config:
    """Configuration class for accessing environment variables with defaults"""

    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Directories
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
    DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "./downloads")

    # Service configuration
    DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Model configuration
    LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_TEXT_MODEL)
    STT_MODEL = os.getenv("STT_MODEL", DEFAULT_SPEECH_MODEL)

    # Create required directories
    @classmethod
    def setup(cls):
        """Set up the environment, creating required directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.DOWNLOAD_DIR, exist_ok=True)

        # Validate configured models
        if cls.LLM_MODEL not in GROQ_TEXT_MODELS:
            print(
                f"Warning: LLM_MODEL {cls.LLM_MODEL} not found in available Groq text models. Using default: {DEFAULT_TEXT_MODEL}"
            )
            cls.LLM_MODEL = DEFAULT_TEXT_MODEL

        if cls.STT_MODEL not in GROQ_SPEECH_MODELS:
            print(
                f"Warning: STT_MODEL {cls.STT_MODEL} not found in available Groq speech models. Using default: {DEFAULT_SPEECH_MODEL}"
            )
            cls.STT_MODEL = DEFAULT_SPEECH_MODEL

    @classmethod
    def get_dict(cls) -> Dict[str, Any]:
        """Return configuration as a dictionary"""
        return {
            "GROQ_API_KEY": cls.GROQ_API_KEY is not None,  # Don't expose actual key
            "OPENAI_API_KEY": cls.OPENAI_API_KEY is not None,  # Don't expose actual key
            "OUTPUT_DIR": cls.OUTPUT_DIR,
            "DOWNLOAD_DIR": cls.DOWNLOAD_DIR,
            "DEBUG": cls.DEBUG,
            "LOG_LEVEL": cls.LOG_LEVEL,
            "LLM_MODEL": cls.LLM_MODEL,
            "STT_MODEL": cls.STT_MODEL,
            "LLM_MODEL_INFO": GROQ_TEXT_MODELS.get(cls.LLM_MODEL),
            "STT_MODEL_INFO": GROQ_SPEECH_MODELS.get(cls.STT_MODEL),
        }

    @classmethod
    def check_api_keys(cls) -> bool:
        """Check if required API keys are present"""
        return cls.GROQ_API_KEY is not None

    @classmethod
    def check_optional_features(cls) -> Dict[str, bool]:
        """Check which optional features are available based on API keys"""
        return {
            "rag": cls.OPENAI_API_KEY is not None,
            "translation": cls.STT_MODEL in ["whisper-large-v3"]
            and cls.GROQ_API_KEY is not None,
        }

    @classmethod
    def get_available_models(cls) -> Dict[str, List[Dict[str, Any]]]:
        """Return available models for the API"""
        return {
            "text_models": [
                {"id": model_id, **model_info}
                for model_id, model_info in GROQ_TEXT_MODELS.items()
            ],
            "speech_models": [
                {"id": model_id, **model_info}
                for model_id, model_info in GROQ_SPEECH_MODELS.items()
            ],
        }
