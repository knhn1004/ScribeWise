"""
Imaging Service for ScribeWise
- Downloads videos 
- Analyzes scenes
- Splits video into chunks for LLM vision models
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import yt_dlp
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torchvision
from PIL import Image
import tempfile
import logging
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImagingService:
    def __init__(self, output_dir: str = "downloads"):
        """Initialize the Imaging Service"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure PyTorch model directory is set and exists
        torch_home = os.environ.get('TORCH_HOME', '/app/torch_cache')
        os.makedirs(torch_home, exist_ok=True)
        os.environ['TORCH_HOME'] = torch_home
        
        # Load the model for image feature extraction
        try:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
            self.model.eval()
            logger.info(f"Loaded ResNet50 model on {device}")
        except Exception as e:
            logger.error(f"Error loading ResNet50 model: {str(e)}")
            raise
        
        # Remove the last fully connected layer to get features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Define preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Threshold for scene change detection
        self.scene_threshold = 0.5
    
    async def download_video(self, url: str) -> Dict[str, str]:
        """Download a video from URL and return file paths"""
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': f'{self.output_dir}/%(id)s.%(ext)s',
            'noplaylist': True,
            'quiet': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = f"{self.output_dir}/{info['id']}.mp4"
                return {
                    "video_path": video_path,
                    "video_id": info['id'],
                    "title": info.get('title', 'Unknown'),
                    "duration": info.get('duration', 0)
                }
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")
    
    async def extract_frames(self, video_path: str, interval: int = 1) -> List[np.ndarray]:
        """Extract frames from video at specified intervals"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps * interval)
        
        count = 0
        success = True
        
        while success:
            success, frame = cap.read()
            if not success:
                break
                
            if count % frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            count += 1
                
        cap.release()
        return frames
    
    async def compute_frame_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Compute embeddings for frames using the ResNet model"""
        features = []
        
        for frame in frames:
            # Preprocess the frame
            frame_tensor = self.transform(frame).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                feature = self.model(frame_tensor)
                feature = feature.squeeze().cpu()
                features.append(feature)
                
        return torch.stack(features)
    
    async def detect_scene_changes(self, 
                                  features: torch.Tensor, 
                                  threshold: float = 0.7) -> List[int]:
        """Detect scene changes based on cosine similarity of features"""
        scene_changes = [0]  # First frame is always a scene change
        
        for i in range(1, len(features)):
            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                features[i-1].unsqueeze(0), 
                features[i].unsqueeze(0)
            )
            
            # If similarity is below threshold, it's a new scene
            if cos_sim < threshold:
                scene_changes.append(i)
                
        return scene_changes
    
    async def process_video(self, url: str) -> Dict:
        """Complete process of downloading and analyzing video scenes"""
        # Download the video
        video_info = await self.download_video(url)
        video_path = video_info["video_path"]
        
        # Extract frames
        frames = await self.extract_frames(video_path)
        
        # Compute features
        features = await self.compute_frame_features(frames)
        
        # Detect scene changes
        scene_changes = await self.detect_scene_changes(features)
        
        # Extract key frames for each scene
        key_frames = []
        scene_times = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for scene_idx in scene_changes:
            frame_time = scene_idx / fps
            scene_times.append(frame_time)
            
            # Save key frame for this scene
            frame_path = f"{self.output_dir}/{video_info['video_id']}_scene_{len(key_frames)}.jpg"
            cv2.imwrite(frame_path, cv2.cvtColor(frames[scene_idx], cv2.COLOR_RGB2BGR))
            key_frames.append(frame_path)
            
        cap.release()
        
        return {
            "video_info": video_info,
            "scene_times": scene_times,
            "key_frames": key_frames,
            "total_scenes": len(scene_changes)
        } 