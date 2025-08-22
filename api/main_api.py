#!/usr/bin/env python3
"""
FastAPI Soccer Action Spotting API
Processes 2-hour videos and returns clips with real-time progress updates
"""
import asyncio
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import yaml
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

# Import core modules
from core.models import ProcessingStatus, ClipInfo
from core.utils import get_clip_info_from_directory, find_clip_file, setup_logging, validate_job_and_get_clips_dir, validate_and_process_video_input
from core.services import JobManager, ProcessingService

# Import inference pipeline
import sys
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
from inference.parallel_inference import run_inference_pipeline_parallel

# Gradio demo runs separately - no imports needed

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Soccer Action Spotting API",
    description="Process soccer videos and generate action clips",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
job_manager = JobManager()
processing_service = ProcessingService(job_manager)

# Skip Gradio mounting for now - access via standalone mode
# To use demo: python3 api/gradio_demo.py



async def process_video_async(job_id: str, video_path: Path):
    """Async video processing with progress updates"""
    start_time = time.perf_counter()
    
    try:
        job_manager.update_job(job_id, status="processing", progress=0.1, message="Starting video processing...")
        
        default_config_path = Path("inference/inference_config.yaml")
        job_manager.update_job(job_id, progress=0.15, message="Using default configuration")
        
        job_manager.update_job(job_id, progress=0.2, message="Running AI models...")
        
        clips_dir = await asyncio.get_event_loop().run_in_executor(
            None, run_inference_pipeline_parallel, video_path, default_config_path
        )
        
        job_manager.update_job(job_id, progress=0.9, message="Finalizing clips...")
        
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
        with open(video_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        logger.info(f"Video uploaded: {file.filename} ({len(content)} bytes)")
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
    cfg = load_config()

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
