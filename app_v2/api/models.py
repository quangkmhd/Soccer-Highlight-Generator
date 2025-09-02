"""
Pydantic models for video management
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class VideoUploadRequest(BaseModel):
    """Request model for video upload"""
    input_path: Optional[str] = Field(None, description="Local file path to video")

class VideoUploadResponse(BaseModel):
    """Response model for video upload"""
    video_id: str
    status: str = "uploaded"
    message: Optional[str] = None

class VideoMetadataResponse(BaseModel):
    """Response model for video metadata"""
    video_id: str
    filename: str
    duration: Optional[float] = Field(None, description="Duration in seconds")
    fps: Optional[float] = Field(None, description="Frames per second")
    resolution: Optional[str] = Field(None, description="Video resolution (e.g., 1920x1080)")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    format: Optional[str] = Field(None, description="Video format (e.g., H.264)")
    file_path: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

class ProcessingRequest(BaseModel):
    """Request model for processing job"""
    pass

class ProcessingResponse(BaseModel):
    """Response model for processing job submission"""
    job_id: str
    status: str = "queued"
    position: Optional[int] = Field(None, description="Position in queue")
    message: Optional[str] = None

class JobStatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    video_id: str
    status: str
    progress_message: str
    position: Optional[int] = Field(None, description="Queue position if queued")
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result_path: Optional[str] = None

class QueueInfoResponse(BaseModel):
    """Response model for queue information"""
    queue_length: int
    current_job: Optional[str] = None
    processing: bool
    queued_jobs: list

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str

# Results & Downloads Models
class HighlightClip(BaseModel):
    """Model for a single highlight clip"""
    clip_id: str
    start: str = Field(description="Start timestamp in format HH:MM:SS.mmm")
    end: str = Field(description="End timestamp in format HH:MM:SS.mmm")
    label: str = Field(description="Action label with score")
    score: int = Field(description="Confidence score 0-100")
    preview_url: str = Field(description="URL to preview the clip")

class ResultsResponse(BaseModel):
    """Response model for job results"""
    job_id: str
    clips: List[HighlightClip]

class ClipSelectionRequest(BaseModel):
    """Request model for clip selection"""
    job_id: str
    clips: List[str] = Field(description="List of clip_ids to select")

class ClipSelectionResponse(BaseModel):
    """Response model for clip selection"""
    job_id: str
    selected: List[str]
