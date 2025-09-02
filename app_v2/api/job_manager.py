"""
Job Manager for processing pipeline
Handles queue, semaphore, and background processing
"""
import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any

class JobInProgressError(Exception):
    """Custom exception raised when a job is already in progress."""
    pass
from pathlib import Path
import logging

from .ai_processing_service import ai_processing_service

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class Job:
    def __init__(self, job_id: str, video_id: str, video_path: str):
        self.job_id = job_id
        self.video_id = video_id
        self.video_path = video_path
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.result_path: Optional[str] = None
        self.progress_message: str = "Waiting in queue"

class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(1)  # Only 1 concurrent processing
        self._processing_task: Optional[asyncio.Task] = None
        self._current_job_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._job_active = False
        
    async def start_processor(self):
        """Start the background job processor"""
        if self._processing_task is None or self._processing_task.done():
            self._processing_task = asyncio.create_task(self._process_jobs())
            logger.info("Job processor started")
    
    async def stop_processor(self):
        """Stop the background job processor"""
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            logger.info("Job processor stopped")
    
    async def submit_job(self, video_id: str, video_path: str) -> str:
        """Submit a new processing job"""
        async with self._lock:
            if self._job_active:
                raise JobInProgressError("AI is busy. Please try again later.")
            self._job_active = True

        job_id = f"job_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
        
        job = Job(job_id, video_id, video_path)
        self._jobs[job_id] = job
        
        await self._queue.put(job_id)
        
        logger.info(f"Job {job_id} submitted for video {video_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and details"""
        job = self._jobs.get(job_id)
        if not job:
            return None
            
        # Since we only allow 1 job at a time, queue position is always 0 or N/A
        queue_position = 0
        
        status_response = {
            "job_id": job_id,
            "video_id": job.video_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "progress_message": job.progress_message
        }
        
        
        if job.started_at:
            status_response["started_at"] = job.started_at.isoformat()
            
        if job.completed_at:
            status_response["completed_at"] = job.completed_at.isoformat()
            
        if job.error_message:
            status_response["error"] = job.error_message
            
        if job.result_path:
            status_response["result_path"] = job.result_path
            
        return status_response
    
    async def _get_queue_position(self, job_id: str) -> int:
        """Get position of job in queue (0-indexed)"""
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.QUEUED:
            return 0  # Not in queue or already processing
        return 0  # Since we only allow 1 job, position is always 0
    
    async def get_queue_info(self) -> Dict[str, Any]:
        """Get current queue information"""
        queued_jobs = [job for job in self._jobs.values() if job.status == JobStatus.QUEUED]
        
        return {
            "queue_length": len(queued_jobs),
            "current_job": self._current_job_id,
            "processing": self._current_job_id is not None,
            "queued_jobs": [
                {
                    "job_id": job.job_id,
                    "video_id": job.video_id,
                    "created_at": job.created_at.isoformat()
                }
                for job in sorted(queued_jobs, key=lambda x: x.created_at)
            ]
        }
    
    async def _process_jobs(self):
        """Background task to process jobs from queue"""
        while True:
            try:
                # Get next job from queue
                job_id = await self._queue.get()
                job = self._jobs.get(job_id)
                
                if not job or job.status != JobStatus.QUEUED:
                    continue
                
                try:
                    async with self._semaphore:
                        self._current_job_id = job_id
                        await self._execute_job(job)
                        self._current_job_id = None
                finally:
                    # Signal that the job slot is now free
                    async with self._lock:
                        self._job_active = False
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in job processor: {e}")
                if job_id and job_id in self._jobs:
                    job = self._jobs[job_id]
                    job.status = JobStatus.FAILED
                    job.error_message = f"Processing error: {str(e)}"
                    job.completed_at = datetime.now()
                self._current_job_id = None
    
    async def _execute_job(self, job: Job):
        """Execute a single job using AI Processing Service"""
        try:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now()
            job.progress_message = "Processing. Please wait around 10-20 minutes."
            
            logger.info(f"Processing job {job.job_id} for video {job.video_id}")
            
            # Use AI Processing Service for core functionality
            result = await ai_processing_service.process_video(
                video_path=job.video_path,
                job_id=job.job_id,
                video_id=job.video_id
            )
            
            job.status = JobStatus.COMPLETED
            job.result_path = result["output_dir"]
            job.progress_message = result["message"]
            job.completed_at = datetime.now()
            
            logger.info(f"Job {job.job_id} completed successfully. Output: {result['output_dir']}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.progress_message = f"Processing failed: {str(e)}"
            job.completed_at = datetime.now()
            
            logger.error(f"Job {job.job_id} failed: {e}")
    

# Global job manager instance
job_manager = JobManager()
