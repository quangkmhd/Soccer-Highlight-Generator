"""Simplified video database operations"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio

from .config import get_config
config = get_config()
# Global variables for simplified access
_db_lock = asyncio.Lock()
_metadata_file = None

def _get_metadata_file() -> Path:
    """Get metadata file path"""
    global _metadata_file
    if _metadata_file is None:
        _metadata_file = Path(__file__).parent.parent / config['storage']['video_metadata_file']
    return _metadata_file

def _load_data() -> Dict[str, Any]:
    """Load data from JSON file - synchronous for simplicity"""
    metadata_file = _get_metadata_file()
    if not metadata_file.exists():
        return {"videos": {}}
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def _save_data(data: Dict[str, Any]) -> None:
    """Save data to JSON file - synchronous for simplicity"""
    metadata_file = _get_metadata_file()
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

async def create_video_record(video_data: Dict[str, Any]) -> str:
    """Create new video record and return video_id"""
    async with _db_lock:
        video_id = str(uuid.uuid4())
        
        data = _load_data()
        data["videos"][video_id] = {
            **video_data,
            "video_id": video_id,
            "created_at": datetime.now(),
            "status": "uploaded"
        }
        
        _save_data(data)
        return video_id

async def get_video_metadata(video_id: str) -> Optional[Dict[str, Any]]:
    """Get video metadata by video_id"""
    async with _db_lock:
        data = _load_data()
        return data["videos"].get(video_id)

async def update_video_metadata(video_id: str, updates: Dict[str, Any]) -> bool:
    """Update video metadata"""
    async with _db_lock:
        data = _load_data()
        if video_id in data["videos"]:
            data["videos"][video_id].update(updates)
            data["videos"][video_id]["updated_at"] = datetime.now()
            _save_data(data)
            return True
        return False

async def list_videos() -> List[Dict[str, Any]]:
    """List all videos"""
    async with _db_lock:
        data = _load_data()
        return list(data["videos"].values())

async def init_db() -> None:
    """Initialize database"""
    metadata_file = _get_metadata_file()
    # Ensure metadata file directory exists
    metadata_file.parent.mkdir(exist_ok=True, parents=True)
