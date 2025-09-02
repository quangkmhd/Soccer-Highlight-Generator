"""
Simplified video operations - consolidated from service layer
"""
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import json

from fastapi import UploadFile, HTTPException
from .config import get_config, get_upload_dir
from . import database

# Global config for simplified access
config = get_config()
upload_dir = get_upload_dir()

async def upload_video_file(file: UploadFile) -> str:
    """Upload video file and return video_id"""
    # Validate file format
    if not file.filename.lower().endswith('.mp4'):
        raise HTTPException(
            status_code=400, 
            detail="Only MP4 format is supported"
        )
    
    # Prepare file path
    file_path = upload_dir / file.filename
    counter = 1
    original_stem = file_path.stem
    original_suffix = file_path.suffix
    
    # Handle duplicate filenames
    while file_path.exists():
        file_path = upload_dir / f"{original_stem}_{counter}{original_suffix}"
        counter += 1
    
    # Stream file to disk to avoid loading entire file into memory
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    max_size = config['video']['max_file_size_mb'] * 1024 * 1024
    
    try:
        with open(file_path, 'wb') as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                
                # Check file size during upload
                file_size += len(chunk)
                if file_size > max_size:
                    # Clean up partial file and raise error
                    buffer.close()
                    file_path.unlink()
                    raise HTTPException(
                        status_code=400,
                        detail=f"File too large. Maximum size is {config['video']['max_file_size_mb']}MB"
                    )
                
                buffer.write(chunk)
    except Exception as e:
        # Clean up on any error
        if file_path.exists():
            file_path.unlink()
        raise
    
    # Get video metadata and validate it
    metadata = await _get_video_info(file_path)
    await _validate_video_properties(metadata, file_path_to_delete=file_path)
    
    # Create database record
    video_data = {
        "filename": file_path.name,
        "file_path": str(file_path),
        "size_bytes": file_size,
        **metadata
    }
    
    video_id = await database.create_video_record(video_data)
    return video_id


async def register_video_path(input_path: str) -> str:
    """Register video from local path without copying - use original path directly"""
    source_path = Path(input_path)
    
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not source_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    # Validate file format
    if not source_path.suffix.lower() == '.mp4':
        raise HTTPException(
            status_code=400, 
            detail="Only MP4 format is supported"
        )
    
    # Check file size
    file_size = source_path.stat().st_size
    max_size = config['video']['max_file_size_mb'] * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {config['video']['max_file_size_mb']}MB"
        )
    
    # Get video metadata and validate it
    metadata = await _get_video_info(source_path)
    await _validate_video_properties(metadata) # No file to delete for registered paths
    
    # Create database record with original path (no copying)
    video_data = {
        "filename": source_path.name,
        "file_path": str(source_path),  # Use original path directly
        "size_bytes": file_size,
        **metadata
    }
    
    video_id = await database.create_video_record(video_data)
    return video_id

def _test_framecv_compatibility(video_path: Path) -> bool:
    """Test if FrameCV can actually read this video file."""
    try:
        import cv2
        # Try to open video with cv2.VideoCapture (same as FrameCV)
        vidcap = cv2.VideoCapture(str(video_path))
        
        # Check if video opened successfully
        if not vidcap.isOpened():
            return False
            
        # Try to read first frame
        ret, frame = vidcap.read()
        vidcap.release()
        
        # Return True if we successfully read a frame
        return ret and frame is not None
        
    except Exception:
        return False

async def _validate_video_properties(metadata: Dict[str, Any], file_path_to_delete: Optional[Path] = None):
    """Validate video duration and FrameCV compatibility, clean up file on failure if path is provided."""
    # Validate video duration
    max_duration = config['video']['max_duration_hours'] * 3600
    if metadata.get('duration', 0) > max_duration:
        if file_path_to_delete:
            file_path_to_delete.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Video too long. Maximum duration is {config['video']['max_duration_hours']} hours"
        )

    # Test actual FrameCV compatibility instead of static codec list
    if file_path_to_delete and not _test_framecv_compatibility(file_path_to_delete):
        file_path_to_delete.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Video format not supported by FrameCV. Codec: '{metadata.get('format', 'unknown')}'"
        )

async def _get_video_info(video_path: Path) -> Dict[str, Any]:
    """Extract video metadata using ffprobe"""
    try:
        # Use ffprobe to get video information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Fallback to basic file info
            return {
                "duration": None,
                "fps": None,
                "resolution": None,
                "format": None
            }
        
        data = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        format_info = data.get('format', {})
        
        metadata = {
            "duration": float(format_info.get('duration', 0)),
            "format": video_stream.get('codec_name', 'unknown') if video_stream else None
        }
        
        if video_stream:
            # Safely parse frame rate
            frame_rate_str = video_stream.get('r_frame_rate', '0/1')
            try:
                if '/' in frame_rate_str:
                    numerator, denominator = frame_rate_str.split('/')
                    fps = float(numerator) / float(denominator) if float(denominator) != 0 else 0
                else:
                    fps = float(frame_rate_str)
            except (ValueError, ZeroDivisionError):
                fps = None
            
            metadata.update({
                "fps": fps,
                "resolution": f"{video_stream.get('width', 0)}x{video_stream.get('height', 0)}"
            })
        
        return metadata
        
    except Exception as e:
        # Fallback to basic file info
        return {
            "duration": None,
            "fps": None,
            "resolution": None,
            "format": None
        }
