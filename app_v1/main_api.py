#!/usr/bin/env python3
"""
FastAPI Soccer Action Spotting API
"""
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import yaml
import uvicorn
import io
import zipfile
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
from contextlib import asynccontextmanager

# Import core modules
from core.models import ProcessingStatus, ClipInfo
from core.utils import get_clip_info_from_directory, find_clip_file, setup_logging, validate_job_and_get_clips_dir, validate_and_process_video_input
from core.services import JobManager, ProcessingService
from core.srt_export import generate_srt_from_clips

# Import inference pipeline
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from inference.parallel_inference import run_inference_pipeline_parallel

# Gradio demo runs separately - no imports needed

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize services
job_manager = JobManager()
processing_service = ProcessingService(job_manager)


# Lifespan handler to replace deprecated on_event shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup section (no-op; logging already configured above)
    yield
    job_manager.cleanup_job(jid)

# Create FastAPI app with lifespan
app = FastAPI(
    title="Soccer Action Spotting API",
    description="Process soccer videos and generate action clips",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def process_video_async(job_id: str, video_path: Path):
    """Async video processing"""
    start_time = time.perf_counter()
    try:
        job_manager.update_job(job_id, status="processing", message="Processing video...")
        
        default_config_path = Path("inference/inference_config.yaml")
        
        clips_dir = await asyncio.get_event_loop().run_in_executor(
            None, run_inference_pipeline_parallel, video_path, default_config_path
        )
        
        processing_time = time.perf_counter() - start_time
        processing_service.complete_job(job_id, clips_dir, processing_time)
        
    except Exception as e:
        processing_service.fail_job(job_id, str(e))


@app.post("/upload", response_model=Dict[str, str])
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    video_path: str = Form(None)
):
    """Upload video file or specify video path and start processing"""
    
    video_file_path, video_name, job_id = validate_and_process_video_input(job_manager, file, video_path)
    
    if file:
        # Stream to disk to avoid loading the entire file into memory
        total_written = 0
        chunk_size = 1024 * 1024  # 1MB
        with open(video_file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                total_written += len(chunk)
        # Ensure job metadata reflects the actual saved path for the uploaded file
        job_manager.update_job(job_id, video_path=str(video_file_path))
        logger.info(f"Video uploaded: {file.filename} ({total_written} bytes) -> {video_file_path}")
    else:
        logger.info(f"Using video path: {video_path}")
    
    background_tasks.add_task(process_video_async, job_id, video_file_path)
    return {"job_id": job_id, "message": "Video uploaded, processing started"}


@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_processing_status(job_id: str):
    """Get real-time processing status"""
    
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ProcessingStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        message=job_data["message"],
        clips_count=job_data.get("clips_count"),
        processing_time=job_data.get("processing_time"),
        error=job_data.get("error")
    )


@app.get("/clips/{job_id}")
async def list_clips(job_id: str) -> List[ClipInfo]:
    """List all generated clips for a job"""
    clips_dir = validate_job_and_get_clips_dir(job_manager, job_id)
    return get_clip_info_from_directory(clips_dir)


@app.get("/clips/{job_id}/gallery")
async def get_clips_gallery(job_id: str) -> List[Dict[str, Any]]:
    """Get clips with metadata for web gallery display"""
    clips_dir = validate_job_and_get_clips_dir(job_manager, job_id)
    clips_info = get_clip_info_from_directory(clips_dir)
    
    return [
        {
            "filename": clip.filename,
            "stream_url": f"/stream/{job_id}/{clip.filename}",
        }
        for clip in clips_info
    ]


@app.get("/stream/{job_id}/{clip_filename}")
async def stream_clip(job_id: str, clip_filename: str):
    """Stream a specific clip file with range support for web players"""
    clips_dir = validate_job_and_get_clips_dir(job_manager, job_id)
    
    clip_path = find_clip_file(clips_dir, clip_filename)
    if not clip_path:
        raise HTTPException(status_code=404, detail="Clip not found")
    
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        filename=clip_filename,
        headers={"Accept-Ranges": "bytes"}
    )


@app.post("/download/{job_id}/zip")
async def download_selected_clips(job_id: str, filenames: List[str]):
    """Download selected clips as a single ZIP archive.

    Parameters
    ----------
    job_id : str
        Processing job identifier.
    filenames : List[str]
        List of clip filenames to include.
    """
    clips_dir = validate_job_and_get_clips_dir(job_manager, job_id)

    # Create in-memory zip
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in filenames:
            clip_path = find_clip_file(clips_dir, fname)
            if clip_path is None:
                continue
            zf.write(clip_path, arcname=fname)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_clips.zip"'}
    )


@app.post("/download/{job_id}/srt")
async def download_selected_srt(job_id: str, filenames: List[str]):
    """Download SRT subtitle file for selected clips.

    Parameters
    ----------
    job_id : str
        Processing job identifier.
    filenames : List[str]
        List of clip filenames to include in SRT.
    """
    clips_dir = validate_job_and_get_clips_dir(job_manager, job_id)
    
    # Validate selected filenames belong to this job
    available_clips = get_clip_info_from_directory(clips_dir)
    available_filenames = {clip.filename for clip in available_clips}
    
    # Filter to only include valid filenames (security measure)
    valid_filenames = [fname for fname in filenames if fname in available_filenames]
    
    if not valid_filenames:
        raise HTTPException(status_code=400, detail="No valid clips selected")
    
    # Get job info for video name
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Extract video name from path
    video_path = Path(job_data.get("video_path", ""))
    video_name = video_path.stem if video_path else job_id
    
    # Generate SRT file in project-local temp directory
    temp_dir = root_dir / "temp_dir" / f"srt_{job_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        srt_path = generate_srt_from_clips(valid_filenames, video_name, temp_dir)
        if not srt_path or not srt_path.exists():
            raise HTTPException(status_code=500, detail="Failed to generate SRT file")
        
        return FileResponse(
            srt_path,
            media_type="text/plain",
            filename=f"{video_name}_highlights.srt",
            headers={"Content-Disposition": f'attachment; filename="{video_name}_highlights.srt"'}
        )
    except Exception as e:
        logger.error(f"Error generating SRT for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate SRT file")


@app.get("/download/{job_id}/srt/all")
async def download_all_srt(job_id: str):
    """Download SRT subtitle file for all clips in a job.

    Parameters
    ----------
    job_id : str
        Processing job identifier.
    """
    clips_dir = validate_job_and_get_clips_dir(job_manager, job_id)
    
    # Get all clips for this job
    available_clips = get_clip_info_from_directory(clips_dir)
    if not available_clips:
        raise HTTPException(status_code=404, detail="No clips found for this job")
    
    all_filenames = [clip.filename for clip in available_clips]
    
    # Get job info for video name
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Extract video name from path
    video_path = Path(job_data.get("video_path", ""))
    video_name = video_path.stem if video_path else job_id
    
    # Generate SRT file in project-local temp directory
    temp_dir = root_dir / "temp_dir" / f"srt_{job_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        srt_path = generate_srt_from_clips(all_filenames, video_name, temp_dir)
        if not srt_path or not srt_path.exists():
            raise HTTPException(status_code=500, detail="Failed to generate SRT file")
        
        return FileResponse(
            srt_path,
            media_type="text/plain",
            filename=f"{video_name}_highlights.srt",
            headers={"Content-Disposition": f'attachment; filename="{video_name}_highlights.srt"'}
        )
    except Exception as e:
        logger.error(f"Error generating SRT for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate SRT file")


@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up temporary files for a job"""
    
    if not job_manager.cleanup_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {"message": f"Job {job_id} cleaned up successfully"}


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Soccer Action Spotting API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "POST /upload",
            "status": "GET /status/{job_id}",
            "clips": "GET /clips/{job_id}",
            "gallery": "GET /clips/{job_id}/gallery",
            "stream": "GET /stream/{job_id}/{clip_filename}",
            "cleanup": "DELETE /cleanup/{job_id}",
        }
    }


def load_config() -> dict:
    """Load server configuration"""
    config_path = Path(__file__).parent / "config_api.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main() -> None:
    """Run the FastAPI server using configuration if available."""
    cfg1 = load_config()
    cfg= cfg1["server"]
    # Use import string to support workers>1
    uvicorn.run(
        "api.main_api:app",
        host=cfg["host"],
        port=cfg["port"],
        workers=cfg["workers"],
        reload=cfg["reload"],
    )


if __name__ == "__main__":
    main()
