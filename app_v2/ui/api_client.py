"""
API Client for Soccer Highlight Detection
"""
import aiohttp
import os
import tempfile
import shutil
from typing import Tuple, List, Dict, Optional
from app_v2.api.config import get_api_base_url


class SoccerHighlightApp:
    def __init__(self):
        self.current_video_name = None
        self.current_video_id = None
        self.current_job_id = None
        self.selected_clips = set()
        self.api_base = get_api_base_url()
        
    async def upload_file(self, file) -> Tuple[str, str]:
        """Upload video file to API"""
        if file is None:
            return "", "Vui lòng chọn file video"

        try:
            async with aiohttp.ClientSession() as session:
                with open(file, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename=os.path.basename(file))

                    async with session.post(f"{self.api_base}/upload/file", data=data) as response:
                        if response.status == 200:
                            result = await response.json()
                            self.current_video_id = result["video_id"]
                            self.current_video_name = os.path.splitext(os.path.basename(file))[0]
                            return result["video_id"], f"{result['message']}"
                        else:
                            error = await response.text()
                            return "", f"Lỗi upload: {error}"
        except Exception as e:
            return "", f"Lỗi: {str(e)}"
    
    async def register_path(self, path: str) -> Tuple[str, str]:
        """Register video path with API"""
        if not path.strip():
            return "", "Vui lòng nhập đường dẫn video"

        try:
            async with aiohttp.ClientSession() as session:
                data = {"input_path": path}
                async with session.post(f"{self.api_base}/upload/path", data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.current_video_id = result["video_id"]
                        self.current_video_name = os.path.splitext(os.path.basename(path))[0]
                        return result["video_id"], f"{result['message']}"
                    else:
                        error = await response.text()
                        return "", f"Lỗi đăng ký: {error}"
        except Exception as e:
            return "", f"Lỗi: {str(e)}"
    
    async def start_processing(self, video_id: str) -> Tuple[Optional[str], str]:
        """Start video processing"""
        if not video_id:
            return None, "❌ Chưa có video để xử lý"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_base}/process/{video_id}") as response:
                    if response.status == 200:
                        result = await response.json()
                        self.current_job_id = result["job_id"]
                        return result["job_id"], f"Bắt đầu xử lý: {result['message']}"
                    else:
                        # Try to parse JSON error detail for cleaner messages
                        try:
                            error_json = await response.json()
                            detail = error_json.get("detail", "Unknown processing error")
                            return None, f"{detail}"
                        except Exception:
                            error_text = await response.text()
                            return None, f"Lỗi xử lý: {error_text}"
        except Exception as e:
            return None, f"Lỗi kết nối: {str(e)}"
    
    async def check_status(self, job_id: str) -> Tuple[str, str, str]:
        """Check job processing status"""
        if not job_id:
            return "Chưa có job", "0%", ""
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base}/status/{job_id}") as response:
                    if response.status == 200:
                        result = await response.json()
                        status = result["status"]
                        progress = result.get("progress", 0)
                        
                        status_text = {
                            "queued": "Đang chờ xử lý",
                            "processing": "Đang xử lý",
                            "completed": "Hoàn thành",
                            "failed": "Thất bại"
                        }.get(status, status)
                        
                        progress_text = f"{progress}%" if progress > 0 else "0%"
                        
                        return status_text, progress_text, status
                    else:
                        return "Lỗi kiểm tra", "0%", "error"
        except Exception as e:
            return f"Lỗi: {str(e)}", "0%", "error"
    
    async def get_results(self, job_id: str) -> Tuple[List[Dict], str]:
        """Get processing results and return clips + HTML display"""
        if not job_id:
            return [], "<p>Chưa có highlights</p>"
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base}/results/{job_id}") as response:
                    if response.status == 200:
                        result = await response.json()
                        clips = result.get("clips", [])
                        return clips, clips
                    else:
                        return [], []
        except Exception as e:
            return [], []
    
    
    async def download_metadata(self, job_id: str, format_type: str, mode: str = "all") -> str:
        """Download metadata (SRT/XML) and return the file path."""
        if not job_id:
            return "❌ Chưa có job để tải"
            
        try:
            async with aiohttp.ClientSession() as session:
                params = {"job_id": job_id, "mode": mode, "format": format_type}
                async with session.get(f"{self.api_base}/download/metadata", params=params) as response:
                    if response.status == 200:
                        # Create a temporary file with the correct extension
                        fd, temp_path = tempfile.mkstemp(suffix=f".{format_type}")
                        with os.fdopen(fd, 'wb') as tmp:
                            content = await response.read()
                            tmp.write(content)
                        return temp_path  # Return the path to the temp file
                    else:
                        error = await response.text()
                        return f"❌ Lỗi tải {format_type.upper()}: {error}"
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"
    
    async def select_clips(self, job_id: str, clip_ids: List[str]) -> Dict:
        """Select clips for download via API"""
        if not job_id:
            raise ValueError("Job ID is required")
            
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "job_id": job_id,
                    "clips": clip_ids
                }
                async with session.post(f"{self.api_base}/select_clips", json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.selected_clips = set(clip_ids)
                        return result
                    else:
                        error = await response.text()
                        raise Exception(f"API error: {error}")
        except Exception as e:
            raise Exception(f"Lỗi kết nối API: {str(e)}")
