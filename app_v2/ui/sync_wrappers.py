"""
Synchronous wrappers for async API client functions
Required for Gradio compatibility
"""
import asyncio
import os
import gradio as gr
from app_v2.ui.api_client import SoccerHighlightApp
from app_v2.api.results_service import results_service
from app_v2.api.config import get_api_base_url

# Create app instance
app = SoccerHighlightApp()
api_base = get_api_base_url()

def _extract_job_id(job_id_input):
    """Extract job_id string from various input formats"""
    if isinstance(job_id_input, (list, tuple)):
        # If it's a list/tuple, take the first non-empty element
        for item in job_id_input:
            if item and isinstance(item, str) and item.startswith('job_'):
                return item
        return None
    elif isinstance(job_id_input, dict):
        # If it's a dict, look for job_id in values
        for value in job_id_input.values():
            if value and isinstance(value, str) and value.startswith('job_'):
                return value
        return None
    elif isinstance(job_id_input, str):
        return job_id_input if job_id_input.startswith('job_') else None
    return None

# --- Auto handlers to hide buttons and trigger upload/register automatically ---
def auto_upload_on_file(file):
    """Automatically upload on file selection and hide the upload button."""
    video_id, status = asyncio.run(app.upload_file(file))
    # Hide the manual upload button once the action is triggered
    return video_id, status

def auto_register_on_path(path):
    """Automatically register path on textbox change and hide the register button.
    Removes absolute paths from the status message so Gradio does not mistake them
    for files/directories and attempt to cache them (which caused IsADirectoryError)."""
    video_id, status = asyncio.run(app.register_path(path))
    # ---- Sanitize status ----
    if status and isinstance(status, str):
        # Replace the full absolute path with just the filename
        status = status.replace(str(path), os.path.basename(str(path)))
        # Also strip current working directory if it sneaks into the message
        status = status.replace(os.getcwd(), '')
        status = status.strip()
    return video_id, status

def start_processing_sync(video_id):
    """Sync wrapper for processing start - also activates timer"""
    # The API client now returns (job_id|None, message)
    job_id, message = asyncio.run(app.start_processing(video_id))
    
    # Activate timer only if processing starts successfully
    timer_update = gr.update(active=True) if job_id else gr.update(active=False)
    
    # The message will contain either the success or error string (e.g., "AI is busy")
    return job_id, message, timer_update

def check_status_sync(job_id):
    """Sync wrapper for status check"""
    job_id = _extract_job_id(job_id)
    if not job_id:
        return "Chưa có job", "0%", ""
    return asyncio.run(app.check_status(job_id))

def get_results_sync(job_id):
    """Sync wrapper for getting results - returns clips, html_display, and selection visibility"""
    job_id = _extract_job_id(job_id)
    if not job_id:
        return [], "<p>Chưa có highlights</p>", gr.update(visible=False)
    
    clips, _ = asyncio.run(app.get_results(job_id))
    
    if not clips:
        return [], "<p>Chưa có highlights</p>", gr.update(visible=False)
    
    # Generate HTML display for clips
    html_content = "<div style='max-height: 400px; overflow-y: auto;'>"
    for i, clip in enumerate(clips):
        clip_id = clip.get('clip_id', f'clip_{i+1}')
        title = clip.get('title', f'Clip {i+1}')
        start_time = clip.get('start_time', '00:00')
        end_time = clip.get('end_time', '00:00')
        score = clip.get('score', 0)
        
        html_content += f"""
        <div style='border: 1px solid #ddd; margin: 5px; padding: 10px; border-radius: 5px;'>
            <h4>{title}</h4>
            <p><strong>Thời gian:</strong> {start_time} - {end_time}</p>
            <p><strong>Điểm số:</strong> {score:.2f}</p>
        </div>
        """
    html_content += "</div>"
    
    return clips, html_content, gr.update(visible=True)


def download_srt_sync(job_id):
    """Generate HTML download link for SRT"""
    job_id = _extract_job_id(job_id)
    if not job_id:
        return '<p style="color: red;">❌ Chưa có job để tải</p>'

    try:
        # Always request selected clips; backend will handle if none selected
        mode = "selected"
        
        download_url = f"{api_base}/download/metadata?job_id={job_id}&mode={mode}&format=srt"
        return f'''
        <div style="margin: 10px 0;">
            <a href="{download_url}" style="
                display: inline-block;
                background-color: #2196F3;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
            ">📝 Tải SRT</a>
            <small style="display: block; margin-top: 5px; color: #666;">
                Mode: {mode} clips
            </small>
        </div>
        '''
    except Exception as e:
        return f'<p style="color: red;">❌ Lỗi tải SRT: {str(e)}</p>'

def download_xml_sync(job_id):
    """Generate HTML download link for XML"""
    job_id = _extract_job_id(job_id)
    if not job_id:
        return '<p style="color: red;">❌ Chưa có job để tải</p>'

    try:
        # Always request selected clips; backend will handle if none selected
        mode = "selected"
        
        download_url = f"{api_base}/download/metadata?job_id={job_id}&mode={mode}&format=xml"
        return f'''
        <div style="margin: 10px 0;">
            <a href="{download_url}" style="
                display: inline-block;
                background-color: #FF9800;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
            ">🔧 Tải XML</a>
            <small style="display: block; margin-top: 5px; color: #666;">
                Mode: {mode} clips
            </small>
        </div>
        '''
    except Exception as e:
        return f'<p style="color: red;">❌ Lỗi tải XML: {str(e)}</p>'

def select_all_clips_sync(job_id):
    """Sync wrapper for selecting all clips"""
    job_id = _extract_job_id(job_id)
    if not job_id:
        return "Chưa có job"
    
    try:
        # Get all available clips
        clips, _ = asyncio.run(app.get_results(job_id))
        if not clips:
            return "Không có clips nào để chọn"
        
        # Extract clip IDs (now using actual filenames)
        clip_ids = [clip.get('clip_id', f'unknown_clip_{i+1}') for i, clip in enumerate(clips)]
        
        # Send selection to API
        result = asyncio.run(app.select_clips(job_id, clip_ids))
        return f"Đã chọn tất cả {len(clip_ids)} clips"
        
    except Exception as e:
        return f"Lỗi khi chọn clips: {str(e)}"

def deselect_all_clips_sync(job_id):
    """Sync wrapper for deselecting all clips"""
    job_id = _extract_job_id(job_id)
    if not job_id:
        return "Chưa có job"
    
    try:
        # Send empty selection to API
        result = asyncio.run(app.select_clips(job_id, []))
        return "Bỏ chọn tất cả clips"
        
    except Exception as e:
        return f"Lỗi khi bỏ chọn clips: {str(e)}"

def sync_clip_selection(job_id, selected_clips_json):
    """Sync wrapper for individual clip selection"""
    if not job_id:
        return "Chưa có job"
    
    try:
        import json
        # Parse the selected clips from JSON string
        if isinstance(selected_clips_json, str):
            selected_clips = json.loads(selected_clips_json)
        else:
            selected_clips = selected_clips_json or []
        
        # Send selection to API
        result = asyncio.run(app.select_clips(job_id, selected_clips))
        return f"Đã cập nhật lựa chọn: {len(selected_clips)} clips"
        
    except Exception as e:
        return f"Lỗi đồng bộ clips: {str(e)}"

def download_selected_clips_sync(job_id, format_choice):
    """Download selected clips metadata in chosen format and return local file path for Gradio File component."""
    job_id = _extract_job_id(job_id)
    if not job_id:
        return gr.update(interactive=False, value=None)  # No job

    try:
        # Determine mode based on whether any clips were selected
        # Always request selected clips; backend will handle if none selected
        mode = "selected"

        fmt = format_choice.lower()  # "srt" or "xml"
        file_path = asyncio.run(app.download_metadata(job_id, fmt, mode=mode))

        # On error, the api_client returns a string starting with "❌"; treat as failure
        if isinstance(file_path, str) and file_path.startswith("❌"):
            return None

        # Build desired filename based on video name
        vid = app.current_video_name or job_id
        ext = format_choice.lower()
        desired_name = f"{vid}_highlights.{ext}"

        # Copy to temp file with desired name so DownloadButton serves correct filename
        import shutil, tempfile, os
        temp_dir = tempfile.mkdtemp()
        dest_path = os.path.join(temp_dir, desired_name)
        shutil.copy(file_path, dest_path)
        # Return update object so caller can assign value and enable button
        return gr.update(value=dest_path, interactive=True)
    except Exception:
        return None


def auto_refresh_status(job_id):
    """Auto refresh status and update results when completed - controls timer state"""
    job_id = _extract_job_id(job_id)
    if not job_id:
        return "Chưa có job", "<p>Chưa có highlights</p>", gr.update(active=False), gr.update(visible=False)
    
    status_text, progress_text, status = check_status_sync(job_id)
    
    # If completed, get results automatically and stop timer
    if status == "completed":
        clips, html_display, selection_visible = get_results_sync(job_id)
        return ("Done", 
                html_display, 
                gr.update(active=False),  # Stop timer when completed
                selection_visible)  # Show selection controls if clips exist
    
    # If failed, show error and stop timer
    if status == "failed":
        return (status_text, 
                "<p>❌ Xử lý thất bại</p>", 
                gr.update(active=False),  # Stop timer when failed
                gr.update(visible=False))  # Hide selection controls
    
    # If queued (waiting), show busy message
    if status == "queued":
        return ("AI đang bận, hãy truy cập vào lúc khác", 
                "<p>⏰ Hệ thống đang bận, vui lòng thử lại sau</p>", 
                gr.update(active=True),  # Keep timer active while queued
                gr.update(visible=False))  # Hide selection controls while queued
    
    # If processing, show processing message
    return ("Đang processing có thể mất từ 15 đến 20p", 
            "<p>⚙️ Đang xử lý video, quá trình có thể mất 15-20 phút</p>", 
            gr.update(active=True),  # Keep timer active while processing
            gr.update(visible=False))  # Hide selection controls while processing
