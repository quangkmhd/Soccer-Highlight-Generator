"""
Utility functions for Soccer Action Spotting API
"""
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import HTTPException
from fastapi import UploadFile
from .models import ClipInfo

logger = logging.getLogger(__name__)


def validate_video_file(filename: str) -> bool:
    """Validate video file extension"""
    return filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))


def get_clip_info_from_directory(clips_dir: Path) -> List[ClipInfo]:
    """Extract clip information from directory"""
    if not clips_dir or not clips_dir.exists():
        return []
    
    clip_files = list(clips_dir.rglob("*.mp4"))
    clips_info = []
    
    for clip_file in clip_files:
        try:
            # Extract basic info - could be enhanced with video metadata parsing
            clips_info.append(ClipInfo(
                filename=clip_file.name))
        except Exception as e:
            logger.warning(f"Failed to process clip {clip_file}: {e}")
    
    return clips_info


def find_clip_file(clips_dir: Path, clip_filename: str) -> Optional[Path]:
    """Find specific clip file in directory"""
    if not clips_dir or not clips_dir.exists():
        return None
    
    clip_files = list(clips_dir.rglob(clip_filename))
    return clip_files[0] if clip_files else None


def validate_job_and_get_clips_dir(job_manager, job_id: str) -> Path:
    """Validate job exists and is completed, return clips directory"""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    clips_dir_str = job_data.get("clips_dir")
    if not clips_dir_str:
        raise HTTPException(status_code=404, detail="Clips directory not found")
    
    clips_dir = Path(clips_dir_str)
    if not clips_dir.exists():
        raise HTTPException(status_code=404, detail="Clips directory not found")
    
    return clips_dir


def validate_and_process_video_input(job_manager, file: UploadFile, video_path: str) -> tuple[Path, str, str]:
    """Validate video input and return video_file_path, video_name, job_id"""
    # Validate input - either file or video_path must be provided
    if not file and not video_path:
        raise HTTPException(status_code=400, detail="Either file upload or video_path must be provided")
    
    if file and video_path:
        raise HTTPException(status_code=400, detail="Provide either file upload OR video_path, not both")
    
    if file:
        if not validate_video_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        video_name = Path(file.filename).stem
        job_id = job_manager.create_job(video_name, Path(file.filename))
        temp_dir = job_manager.get_temp_dir(job_id)
        video_file_path = temp_dir / file.filename
        return video_file_path, video_name, job_id
    else:
        video_file_path = Path(video_path)
        if not video_file_path.exists():
            raise HTTPException(status_code=400, detail=f"Video file not found: {video_path}")
        
        if not validate_video_file(video_file_path.name):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        video_name = video_file_path.stem
        job_id = job_manager.create_job(video_name, video_file_path)
        return video_file_path, video_name, job_id


def setup_logging() -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
