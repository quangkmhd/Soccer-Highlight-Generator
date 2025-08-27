"""
Business logic services for Soccer Action Spotting API
"""
import asyncio
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class JobManager:
    """Manages processing jobs and temporary files"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.temp_dirs: Dict[str, Path] = {}
    
    def create_job(self, video_name: str, video_path: Path, is_upload: bool = False) -> str:
        """Create a new processing job"""
        job_id = str(uuid.uuid4())
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix=f"soccer_api_{job_id}_"))
        self.temp_dirs[job_id] = temp_dir
        
        # Initialize job data
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "uploaded",
            "progress": 0.0,
            "message": "Video ready for processing",
            "video_name": video_name,
            "video_path": str(video_path),
            "is_upload": is_upload
        }
        
        logger.info(f"Created job {job_id} for video: {video_name}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data by ID"""
        return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, **updates) -> None:
        """Update job data"""
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
    
    def cleanup_job(self, job_id: str) -> bool:
        """Clean up job and temporary files"""
        if job_id not in self.jobs:
            return False

        job = self.jobs.get(job_id, {})

        # Remove temporary directory only if it was an upload
        if job.get("is_upload") and job_id in self.temp_dirs:
            temp_dir = self.temp_dirs[job_id]
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
            del self.temp_dirs[job_id]

        # Remove generated clips directory if it exists
        clips_dir_str = job.get("clips_dir")
        if clips_dir_str:
            clips_path = Path(clips_dir_str)
            try:
                if clips_path.exists():
                    import shutil
                    shutil.rmtree(clips_path)
            except Exception as e:
                logger.warning(f"Failed to remove clips dir for job {job_id}: {e}")

        # Remove job from memory
        del self.jobs[job_id]
        logger.info(f"Cleaned up job {job_id}")
        return True
    
    def get_temp_dir(self, job_id: str) -> Optional[Path]:
        """Get temporary directory for job"""
        return self.temp_dirs.get(job_id)


class ProcessingService:
    """Handles video processing pipeline utilities"""
    
    def __init__(self, job_manager: JobManager):
        self.job_manager = job_manager
    
    
    def count_clips(self, clips_dir: Path) -> int:
        """Count generated clips in directory"""
        return len(list(clips_dir.rglob("*.mp4")))
    
    def complete_job(self, job_id: str, clips_dir: Path, processing_time: float) -> None:
        """Mark job as completed with results"""
        clips_count = self.count_clips(clips_dir)
        
        self.job_manager.update_job(
            job_id,
            status="completed",
            progress=1.0,
            message=f"Processing completed in {processing_time:.2f}s",
            clips_count=clips_count,
            clips_dir=str(clips_dir),
            processing_time=processing_time
        )
        
        logger.info(f"Job {job_id} completed: {clips_count} clips generated in {processing_time:.2f}s")
    
    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed with error"""
        logger.error(f"Job {job_id} failed: {error}")
        self.job_manager.update_job(
            job_id,
            status="failed",
            progress=0.0,
            message="Processing failed",
            error=error
        )