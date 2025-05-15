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
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImagingService:
    def __init__(self, output_dir: str = "downloads"):
        """Initialize the Imaging Service"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        torch_home = os.environ.get("TORCH_HOME", "/app/torch_cache")
        os.makedirs(torch_home, exist_ok=True)
        os.environ["TORCH_HOME"] = torch_home

        try:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
            self.model.eval()
            logger.info(f"Loaded ResNet50 model on {device}")
        except Exception as e:
            logger.error(f"Error loading ResNet50 model: {str(e)}")
            raise

        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.scene_threshold = 0.5

    async def download_video(self, url: str) -> Dict[str, str]:
        """Download a video from URL and return file paths"""
        ydl_opts = {
            "format": "bestvideo[vcodec!*=av01][ext=mp4]+bestaudio[ext=m4a]/best[vcodec!*=av01][ext=mp4]/best[ext=mp4]/best",
            "outtmpl": f"{self.output_dir}/%(id)s.%(ext)s",
            "noplaylist": True,
            "quiet": False,
            "recode_video": "mp4",
            "postprocessor_args": {
                "ffmpeg": [
                    "-codec:v",
                    "libx264",
                    "-crf",
                    "23",
                ]
            },
            "retries": 5,
            "fragment_retries": 5,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading video from URL: {url}")
                info = ydl.extract_info(url, download=True)
                video_id = info["id"]

                video_path = f"{self.output_dir}/{video_id}.mp4"
                if not os.path.exists(video_path):
                    for ext in ["mkv", "webm", "mp4"]:
                        alt_path = f"{self.output_dir}/{video_id}.{ext}"
                        if os.path.exists(alt_path):
                            logger.warning(
                                f"Found video with unexpected format: {alt_path}"
                            )
                            video_path = alt_path
                            break

                if not os.path.exists(video_path):
                    raise Exception(f"Downloaded video file not found: {video_path}")

                logger.info(f"Successfully downloaded video to: {video_path}")

                return {
                    "video_path": video_path,
                    "video_id": video_id,
                    "title": info.get("title", "Unknown"),
                    "duration": info.get("duration", 0),
                }
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            raise Exception(f"Error downloading video: {str(e)}")

    async def extract_frames(
        self, video_path: str, interval: int = 1
    ) -> List[np.ndarray]:
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            count += 1

        cap.release()
        return frames

    async def compute_frame_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Compute embeddings for frames using the ResNet model"""
        features = []

        for frame in frames:
            frame_tensor = self.transform(frame).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = self.model(frame_tensor)
                feature = feature.squeeze().cpu()
                features.append(feature)

        return torch.stack(features)

    async def detect_scene_changes(
        self, features: torch.Tensor, threshold: float = 0.7
    ) -> List[int]:
        """Detect scene changes based on cosine similarity of features"""
        scene_changes = [0]

        for i in range(1, len(features)):
            cos_sim = torch.nn.functional.cosine_similarity(
                features[i - 1].unsqueeze(0), features[i].unsqueeze(0)
            )

            if cos_sim < threshold:
                scene_changes.append(i)

        return scene_changes

    async def extract_scenes_pyscenedetect(
        self, video_path: str
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract scene boundaries and key frames using PySceneDetect
        Returns a list of tuples (frame_number, frame_image)
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            raise Exception(f"Video file not found: {video_path}")

        try:
            import subprocess

            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True,
                text=True,
            )
            codec = result.stdout.strip()
            logger.info(f"Video codec detected: {codec}")

            if "av1" in codec.lower():
                logger.warning(
                    f"AV1 codec detected, trying to transcode to H.264 first"
                )
                temp_output = f"{os.path.splitext(video_path)[0]}_h264.mp4"

                transcode_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-preset",
                    "fast",
                    temp_output,
                ]
                subprocess.run(transcode_cmd, check=True)

                if os.path.exists(temp_output):
                    logger.info(
                        f"Successfully transcoded video to H.264: {temp_output}"
                    )
                    video_path = temp_output
                else:
                    logger.warning("Transcoding failed, will try original file anyway")
        except Exception as e:
            logger.warning(f"Failed to check or transcode video: {str(e)}")

        try:
            thresholds = [15.0, 30.0, 45.0]
            scenes = []

            for threshold in thresholds:
                video_manager = VideoManager([video_path])
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=threshold))

                video_manager.set_downscale_factor(factor=2)

                logger.info(f"Starting PySceneDetect with threshold={threshold}")
                video_manager.start()
                scene_manager.detect_scenes(frame_source=video_manager)
                scene_list = scene_manager.get_scene_list()
                video_manager.release()

                logger.info(
                    f"PySceneDetect: threshold={threshold}, scenes found={len(scene_list)}"
                )

                if len(scene_list) > 0:
                    break

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                raise Exception(f"Failed to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"Video properties: {width}x{height} @ {fps}fps")

            if len(scene_list) == 0:
                logger.warning(
                    "No scenes detected, extracting first frame as fallback."
                )

                success, frame = cap.read()
                if success and frame is not None:
                    scenes.append((0, frame))
                    logger.info("Successfully extracted first frame as fallback")
                else:
                    logger.warning(
                        "Failed to read first frame, trying alternative positions"
                    )
                    for pos in [0.1, 0.25, 0.5, 0.75]:
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        target_frame = int(total_frames * pos)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        success, frame = cap.read()
                        if success and frame is not None:
                            scenes.append((target_frame, frame))
                            logger.info(
                                f"Successfully extracted frame at position {pos}"
                            )
                            break
            else:
                for scene in scene_list:
                    frame_num = scene[0].get_frames()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    success, frame = cap.read()
                    if success and frame is not None:
                        scenes.append((frame_num, frame))

            cap.release()

            if len(scenes) == 0:
                logger.error(
                    "No frames could be extracted from the video after all attempts"
                )
                raise Exception("Failed to extract any frames from video")

            logger.info(f"Successfully extracted {len(scenes)} frames from video")
            return scenes

        except Exception as e:
            logger.error(f"Error in scene detection: {str(e)}")
            try:
                logger.warning(
                    "Attempting fixed interval frame extraction as last resort"
                )
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise Exception("Cannot open video even for last resort extraction")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                intervals = [int(total_frames * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]

                scenes = []
                for frame_pos in intervals:
                    if frame_pos <= 0:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    success, frame = cap.read()
                    if success and frame is not None:
                        scenes.append((frame_pos, frame))

                cap.release()

                if len(scenes) > 0:
                    logger.info(f"Last resort extraction found {len(scenes)} frames")
                    return scenes
            except Exception as inner_e:
                logger.error(f"Last resort extraction also failed: {str(inner_e)}")

            raise Exception(f"Scene detection and all fallbacks failed: {str(e)}")

    async def process_video(self, url: str) -> Dict:
        """Process video and extract key frames."""
        try:
            video_info = await self.download_video(url)
            video_path = video_info["video_path"]

            if not os.path.exists(video_path):
                logger.error(f"Downloaded video file not found: {video_path}")
                raise Exception(f"Video file not found: {video_path}")

            logger.info(f"Using PySceneDetect for scene splitting: {video_path}")
            scenes = await self.extract_scenes_pyscenedetect(video_path)

            if not scenes:
                logger.error(
                    "No scenes detected or frames extracted. Check video file and PySceneDetect settings."
                )
                raise Exception("No scenes detected in video.")

            key_frames = []
            scene_times = []
            fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
            video_id = video_info["video_id"]

            video_frames_dir = f"{self.output_dir}/{video_id}"
            os.makedirs(video_frames_dir, exist_ok=True)

            for i, (frame_num, frame) in enumerate(scenes):
                frame_time = frame_num / fps if fps > 0 else 0
                scene_times.append(frame_time)

                frame_path = f"{video_frames_dir}/scene_{i}.jpg"
                cv2.imwrite(frame_path, frame)
                key_frames.append(frame_path)

                logger.info(f"Saved frame {i} to {frame_path}")

            return {
                "video_info": video_info,
                "scene_times": scene_times,
                "key_frames": key_frames,
                "total_scenes": len(scenes),
            }

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise Exception(f"Error processing video: {str(e)}")
