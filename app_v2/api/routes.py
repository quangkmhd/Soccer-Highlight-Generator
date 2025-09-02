"""
API routes for video management
"""
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import FileResponse, StreamingResponse

from .models import (
    VideoUploadResponse, 
    VideoMetadataResponse,
    ProcessingResponse, 
    JobStatusResponse,
    ClipSelectionRequest,
    ClipSelectionResponse,
    QueueInfoResponse,
    ResultsResponse
)
from . import video_service
from .job_manager import job_manager, JobInProgressError
from .results_service import results_service
from . import database

video_router = APIRouter()

@video_router.post("/upload/file", response_model=VideoUploadResponse)
async def upload_video_file(
    file: UploadFile = File(..., description="Video file to upload")
):
    """
    Upload video file
    """
    try:
        video_id = await video_service.upload_video_file(file)
        return VideoUploadResponse(
            video_id=video_id,
            status="uploaded",
            message="Video file uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@video_router.post("/upload/path", response_model=VideoUploadResponse)
async def register_video_path(
    input_path: str = Form(..., description="Local path to video file")
):
    """
    Register local video path
    """
    try:
        video_id = await video_service.register_video_path(input_path)
        return VideoUploadResponse(
            video_id=video_id,
            status="registered", 
            message="Video path registered successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Path registration failed: {str(e)}")


@video_router.get("/video/{video_id}", response_model=VideoMetadataResponse)
async def get_video_metadata(video_id: str):
    """
    Get video metadata by video_id
    Returns duration, fps, resolution, size, format information
    """
    try:
        # Directly get metadata from database - simplified call chain
        metadata = await database.get_video_metadata(video_id)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail="Video not found"
            )
        
        return VideoMetadataResponse(**metadata)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")

@video_router.post("/process/{video_id}", response_model=ProcessingResponse)
async def start_processing(video_id: str):
    """
    Start AI processing for a video
    Only 1 video can be processed at a time
    """
    try:
        # Get video metadata to check if video exists and get file path
        video_metadata = await database.get_video_metadata(video_id)    
        video_path = video_metadata["file_path"]
        
        try:
            # Submit job to processing queue
            job_id = await job_manager.submit_job(video_id, video_path)
            
            job_status = await job_manager.get_job_status(job_id)
            
            return ProcessingResponse(
                job_id=job_id,
                status=job_status["status"],
                message="Wait for processing to complete... (around 15-20 minutes)"
            )
        except JobInProgressError:
            raise HTTPException(
                status_code=409, 
                detail="AI is busy. Please try again later."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@video_router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Poll processing status for a job
    Returns current status and progress information
    """
    try:
        job_status = await job_manager.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )
        
        return JobStatusResponse(**job_status)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@video_router.get("/queue/status", response_model=QueueInfoResponse)
async def get_queue_status():
    """
    Get current queue information
    Shows queue length, current job, and queued jobs
    """
    try:
        queue_info = await job_manager.get_queue_info()
        return QueueInfoResponse(**queue_info)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get queue status: {str(e)}")

# Results & Downloads Endpoints
@video_router.get("/results/{job_id}", response_model=ResultsResponse)
async def get_job_results(job_id: str):
    """
    Get highlight clips for a completed job
    Returns list of clips with timestamps, labels, and scores
    """
    try:
        # Get job status to check if completed and get result path
        job_status = await job_manager.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )
        
        if job_status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed. Current status: {job_status['status']}"
            )
        
        # Get results from the results service
        clips = await results_service.get_job_results(job_id, job_status.get("result_path"))
        
        return ResultsResponse(
            job_id=job_id,
            clips=clips
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")

@video_router.post("/select_clips", response_model=ClipSelectionResponse)
async def select_clips(request: ClipSelectionRequest):
    """
    Save user's clip selection for download
    UI calls this when user checks/unchecks highlight clips
    """
    try:
        # Verify job exists
        job_status = await job_manager.get_job_status(request.job_id)
        if not job_status:
            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )
        
        # Save clip selection
        success = await results_service.save_clip_selection(request.job_id, request.clips)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save clip selection"
            )
        
        return ClipSelectionResponse(
            job_id=request.job_id,
            selected=request.clips
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to select clips: {str(e)}")


@video_router.get("/download/metadata")
async def download_metadata(
    job_id: str = Query(..., description="Job ID to export metadata from"),
    mode: str = Query("selected", description="Export mode: 'selected'"),
    format: str = Query("srt", description="Export format: 'srt' or 'xml'")
):
    """
    Export highlight metadata for video editing software
    UI calls this when user clicks "Export SRT" or "Export XML"
    """
    try:
        # Verify job exists and is completed
        job_status = await job_manager.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed. Current status: {job_status['status']}"
            )
        
        # Get original video filename
        video_metadata = await database.get_video_metadata(job_status["video_id"])
        original_filename = "highlights"
        if video_metadata and "original_filename" in video_metadata:
            original_filename = Path(video_metadata["original_filename"]).stem
        elif video_metadata and "file_path" in video_metadata:
            original_filename = Path(video_metadata["file_path"]).stem
        
        # Get job results
        clips = await results_service.get_job_results(job_id, job_status.get("result_path"))
        if not clips:
            raise HTTPException(status_code=404, detail="No clips found for this job")
        
        # Generate metadata file with base name derived from original video filename
        base_output_name = f"{original_filename}_highlights"
        metadata_path = await results_service.generate_metadata(job_id, clips, format, mode, output_name=base_output_name)
        
        # Determine MIME type based on format
        media_type = "text/plain" if format.lower() == "srt" else "application/xml"
        filename = f"{original_filename}_highlights.{format.lower()}"
        
        # Return file for download
        return FileResponse(
            path=metadata_path,
            media_type=media_type,
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export metadata: {str(e)}")

@video_router.get("/clips/{clip_id:path}")
async def stream_clip_preview(clip_id: str):
    """
    Stream clip preview for highlight clips
    clip_id format: job_id/clip_filename.mp4 (e.g., job_14a2ab5f_1756610853/100_01-23-37_01-25-08_Penalty_Goal.mp4)
    """
    try:
        parts = clip_id.split('/', 1)
        job_id = parts[0]
        clip_filename = parts[1]
        
        # Verify job exists and is completed
        job_status = await job_manager.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_status["status"] != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not completed. Current status: {job_status['status']}"
            )
        
        # Get result path from job status
        result_path = job_status.get("result_path")
        if not result_path or not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Result path not found")
        
        # Find clips directory
        clips_dir = None
        if os.path.isdir(result_path):
            # Look for subdirectory containing video clips
            for item in os.listdir(result_path):
                item_path = os.path.join(result_path, item)
                if os.path.isdir(item_path):
                    clips_dir = item_path
                    break
            
            # If no subdirectory, use the result_path itself
            if not clips_dir:
                clips_dir = result_path
        else:
            clips_dir = os.path.dirname(result_path)
        
        if not clips_dir or not os.path.exists(clips_dir):
            raise HTTPException(status_code=404, detail="Clips directory not found")
        
        # With the new format, clip_id is already the actual filename (without extension)
        # clip_filename includes .mp4 extension, so we can use it directly
        clip_path = os.path.join(clips_dir, clip_filename)
        
        # Verify the file exists
        if not os.path.exists(clip_path):
            # If exact filename doesn't exist, try to find it in the directory
            mp4_files = [f for f in os.listdir(clips_dir) if f.endswith('.mp4')]
            if not mp4_files:
                raise HTTPException(status_code=404, detail="No MP4 files found in clips directory")
            
            # Check if any files match the requested filename
            matching_file = None
            for mp4_file in mp4_files:
                if mp4_file == clip_filename:
                    matching_file = mp4_file
                    break
            
            if not matching_file:
                raise HTTPException(status_code=404, detail=f"Clip file {clip_filename} not found")
            
            clip_path = os.path.join(clips_dir, matching_file)
        
        if not os.path.exists(clip_path):
            raise HTTPException(status_code=404, detail="Clip file not found")
        
        # Stream the video file
        def iter_file(file_path: str):
            with open(file_path, mode="rb") as file_like:
                while True:
                    chunk = file_like.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    yield chunk
        
        file_size = os.path.getsize(clip_path)
        
        return StreamingResponse(
            iter_file(clip_path),
            media_type="video/mp4",
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Content-Disposition": f"inline; filename={clip_filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream clip: {str(e)}")

