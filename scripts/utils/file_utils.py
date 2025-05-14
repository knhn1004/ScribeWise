"""
File utility functions for ScribeWise
"""

import os
import shutil
import json
from typing import Dict, Any, List, Optional
import datetime
from pydantic import HttpUrl


class PydanticJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Pydantic types like HttpUrl"""
    def default(self, obj):
        # Handle HttpUrl objects by converting them to strings
        if isinstance(obj, HttpUrl):
            return str(obj)
        # Let the base class handle other types or raise TypeError
        return super().default(obj)


def ensure_directory(directory: str) -> str:
    """Ensure a directory exists, creating it if necessary"""
    os.makedirs(directory, exist_ok=True)
    return directory


def save_json(data: Dict[str, Any], filepath: str) -> str:
    """Save data as JSON file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=PydanticJSONEncoder)
    
    return filepath


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_text(text: str, filepath: str) -> str:
    """Save text to file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return filepath


def load_text(filepath: str) -> str:
    """Load text from file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


def create_output_directory(base_dir: str = "outputs", video_id: str = None) -> str:
    """Create an output directory for a specific video or with timestamp"""
    if video_id:
        dir_name = f"{base_dir}/{video_id}"
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{base_dir}/{timestamp}"
    
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def cleanup_temp_files(directory: str, extensions: List[str] = None) -> None:
    """Clean up temporary files in a directory"""
    if not os.path.exists(directory):
        return
    
    if extensions:
        for ext in extensions:
            for file in os.listdir(directory):
                if file.endswith(ext):
                    os.remove(os.path.join(directory, file))
    else:
        # Just remove the entire directory
        shutil.rmtree(directory)
        
        
def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return os.path.getsize(filepath) 