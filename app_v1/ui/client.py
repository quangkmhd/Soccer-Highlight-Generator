"""HTTP client for the Soccer Action Spotting API used by the Gradio demo."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .utils import extract_score_from_filename, sort_clips_by_score

logger = logging.getLogger(__name__)


class SoccerAPIClient:
    """Client for Soccer Action Spotting API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def upload_video(self, video_file: Optional[str], video_path: Optional[str]) -> Tuple[bool, str]:
        """Upload a video file or submit a local path.
        Returns (success, job_id_or_error).
        """
        try:
            if video_file and video_path:
                return False, "Please provide either file upload OR video path, not both"
            if not video_file and not video_path:
                return False, "Please provide either a video file or video path"

            if video_file:
                with open(video_file, "rb") as f:
                    files = {"file": (Path(video_file).name, f, "video/mp4")}
                    response = requests.post(f"{self.base_url}/upload", files=files, timeout=60)
            else:
                data = {"video_path": video_path}
                response = requests.post(f"{self.base_url}/upload", data=data, timeout=30)

            if response.status_code == 200:
                job_id = response.json().get("job_id", "")
                return (True, job_id) if job_id else (False, "Invalid response: missing job_id")
            return False, f"Upload failed: {response.text}"
        except Exception as e:
            logger.exception("Error uploading video")
            return False, f"Error uploading video: {e}"

    def get_status(self, job_id: str) -> Tuple[str, float, str, Optional[int]]:
        """Get job status: (status, progress, message, clips_count)."""
        try:
            response = requests.get(f"{self.base_url}/status/{job_id}", timeout=15)
            if response.status_code == 200:
                data = response.json()
                return (
                    data.get("status", "error"),
                    float(data.get("progress", 0.0)),
                    str(data.get("message", "")),
                    data.get("clips_count"),
                )
            return "error", 0.0, f"Status check failed: {response.text}", None
        except Exception as e:
            logger.exception("Error checking status")
            return "error", 0.0, f"Error checking status: {e}", None

    def get_clips_gallery(self, job_id: str) -> List[Tuple[str, str]]:
        """Return list of (url, label) tuples sorted by score descending."""
        try:
            response = requests.get(f"{self.base_url}/clips/{job_id}/gallery", timeout=30)
            if response.status_code != 200:
                return []
            clips = response.json()
            sorted_clips = sort_clips_by_score(clips)
            return [
                (f"{self.base_url}{clip['stream_url']}", extract_score_from_filename(clip["filename"]))
                for clip in sorted_clips
            ]
        except Exception as e:
            logger.error(f"Error fetching clips: {e}")
            return []

    def get_clips_data(self, job_id: str) -> List[Dict[str, Any]]:
        """Return list of dicts: {url, filename, label} sorted by score descending."""
        try:
            response = requests.get(f"{self.base_url}/clips/{job_id}/gallery", timeout=30)
            if response.status_code != 200:
                return []
            clips = response.json()
            sorted_clips = sort_clips_by_score(clips)
            return [
                {
                    "url": f"{self.base_url}{clip['stream_url']}",
                    "filename": clip["filename"],
                    "label": extract_score_from_filename(clip["filename"]),
                }
                for clip in sorted_clips
            ]
        except Exception as e:
            logger.error(f"Error fetching clips data: {e}")
            return []

    def download_clips_zip(self, job_id: str, filenames: List[str]) -> Optional[Path]:
        """Download a ZIP archive of selected clips.
        Returns Path to saved zip file or None on failure.
        """
        if not filenames:
            return None
        try:
            # Use project-local temp directory
            root_dir = Path(__file__).resolve().parents[2]
            base_temp = root_dir / "temp_dir"
            base_temp.mkdir(parents=True, exist_ok=True)
            dest_dir = base_temp / f"soccer_zip_{job_id}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            zip_path = dest_dir / f"{job_id}_clips.zip"
            url = f"{self.base_url}/download/{job_id}/zip"
            response = requests.post(url, json=filenames, stream=True, timeout=300)
            if response.status_code != 200:
                logger.error(f"Zip download failed: {response.text}")
                return None
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return zip_path
        except Exception as e:
            logger.error(f"Error downloading zip: {e}")
            return None

    def download_srt(self, job_id: str, filenames: List[str]) -> Optional[Path]:
        """Download SRT subtitle file for selected clips.
        Returns Path to saved SRT file or None on failure.
        """
        if not filenames:
            return None
        try:
            # Use project-local temp directory
            root_dir = Path(__file__).resolve().parents[2]
            base_temp = root_dir / "temp_dir"
            base_temp.mkdir(parents=True, exist_ok=True)
            dest_dir = base_temp / f"soccer_srt_{job_id}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            url = f"{self.base_url}/download/{job_id}/srt"
            response = requests.post(url, json=filenames, stream=True, timeout=60)
            if response.status_code != 200:
                logger.error(f"SRT download failed: {response.text}")
                return None
            
            # Extract filename from Content-Disposition header
            content_disposition = response.headers.get('content-disposition', '')
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[-1].strip('"')
            else:
                filename = f"{job_id}_highlights.srt"
            
            srt_path = dest_dir / filename
            with open(srt_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return srt_path
        except Exception as e:
            logger.error(f"Error downloading SRT: {e}")
            return None

    def download_all_srt(self, job_id: str) -> Optional[Path]:
        """Download SRT subtitle file for all clips in a job.
        Returns Path to saved SRT file or None on failure.
        """
        try:
            # Use project-local temp directory
            root_dir = Path(__file__).resolve().parents[2]
            base_temp = root_dir / "temp_dir"
            base_temp.mkdir(parents=True, exist_ok=True)
            dest_dir = base_temp / f"soccer_srt_{job_id}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            url = f"{self.base_url}/download/{job_id}/srt/all"
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code != 200:
                logger.error(f"All SRT download failed: {response.text}")
                return None
            
            # Extract filename from Content-Disposition header
            content_disposition = response.headers.get('content-disposition', '')
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[-1].strip('"')
            else:
                filename = f"{job_id}_highlights.srt"
            
            srt_path = dest_dir / filename
            with open(srt_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return srt_path
        except Exception as e:
            logger.error(f"Error downloading all SRT: {e}")
            return None

    def cleanup_job(self, job_id: str) -> bool:
        """Cleanup server-side job files."""
        try:
            response = requests.delete(f"{self.base_url}/cleanup/{job_id}", timeout=20)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error cleaning up job: {e}")
            return False
