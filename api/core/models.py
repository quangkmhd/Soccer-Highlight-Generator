"""
Data models for Soccer Action Spotting API
"""
from typing import Optional, List, Any
from pydantic import BaseModel, Field





class ProcessingStatus(BaseModel):
    """Processing status response"""
    job_id: str
    status: str  # "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    clips_count: Optional[int] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


class ClipInfo(BaseModel):
    """Information about generated clips"""
    filename: str


class JobData(BaseModel):
    """Internal job data structure"""
    job_id: str
    status: str
    progress: float
    message: str
    video_name: str
    video_path: str
    clips_dir: Optional[str] = None
    clips_count: Optional[int] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
