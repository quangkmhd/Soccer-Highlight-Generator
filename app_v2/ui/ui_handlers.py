"""
UI event handlers, moved from interface.py to separate logic from layout.
"""
import gradio as gr
import json
import app_v2.ui.sync_wrappers as sw
from app_v2.api.config import get_config
import os
# --- Client-side validation --- 
def load_js_validation():
    """Validate video file"""
    config = get_config()
    max_duration_seconds = config['video']['max_duration_hours'] * 3600
    max_file_size_bytes = config['video']['max_file_size_mb'] * 1024 * 1024
    
    ui_dir = os.path.dirname(__file__)
    js_path = os.path.join(ui_dir, "js", "validation.js")
    
    with open(js_path, "r") as f:
        js_template = f.read()
        
    js_script = f"""
    <script>
    {js_template}
    document.addEventListener('DOMContentLoaded', (event) => {{
        setupVideoValidation('video-file-input', {max_duration_seconds}, {max_file_size_bytes});
    }});
    </script>
    """
    return js_script

def toggle_clip(checked, selection, clip_id, job_id):
    """Update selection list and sync to backend"""
    selection = list(selection or [])
    if checked and clip_id and clip_id not in selection:
        selection.append(clip_id)
    elif (not checked) and clip_id in selection:
        selection.remove(clip_id)
    # Sync to backend (non-blocking)
    try:
        _ = sw.sync_clip_selection(job_id, json.dumps(selection))
    except Exception:
        pass
    # Enable create-metadata button only when at least one clip selected
    create_btn_update = gr.update(interactive=bool(selection))
    # Reset download button whenever selection changes (disable & clear value)
    download_btn_update = gr.update(interactive=False, value=None)
    return selection, f"Đã chọn: {selection}", create_btn_update, download_btn_update

def refresh(job_id, selection, max_clips):
    """Timer refresh: update status + clips grid"""
    job_id = sw._extract_job_id(job_id)
    if not job_id:
        # No job yet – hide all clips
        updates = ["Chưa có job", f"Đã chọn: {selection}"]
        for _ in range(max_clips):
            updates.extend([gr.update(value=None, visible=False), gr.update(visible=False), None])
        updates.append(gr.update(visible=False))  # selection_controls
        updates.append(gr.update(active=False))   # timer
        return updates

    status_text, _prog, status_code = sw.check_status_sync(job_id)

    # Only request clips from backend when processing is done to prevent 400 errors
    if status_code != "completed":
        # Hide clips until job is finished (avoid calling /results prematurely)
        updates = [status_text, f"Đã chọn: {selection}"]
        for _ in range(max_clips):
            updates.extend([gr.update(value=None, visible=False), gr.update(visible=False), None])
        updates.append(gr.update(visible=False))  # selection_controls
        # Keep timer active while still queued/processing
        timer_active = status_code not in ("failed", "completed")
        updates.append(gr.update(active=timer_active))
        # Buttons state while processing/queued
        updates.append(gr.update(interactive=bool(selection)))  # create button
        updates.append(gr.update(interactive=False, value=None))  # download button
        return updates

    # When completed, load clips once
    clips, _html_display, _sel_visible = sw.get_results_sync(job_id)

    updates = [status_text, f"Đã chọn: {selection}"]
    clip_ids_batch = []
    for i in range(max_clips):
        if i < len(clips):
            clip = clips[i]
            clip_id = clip.get('clip_id')
            video_url = clip.get('preview_url', '')
            score = clip.get('score', 0)
            # Extract event label
            full_label = clip.get('label', '')
            action_label = full_label.split(' ')[-1] if full_label else ''
            # Extract start & end times (strip milliseconds if present)
            start_ts = clip.get('start', '')
            end_ts = clip.get('end', '')
            start_fmt = start_ts.split('.')[0] if start_ts else ''
            end_fmt = end_ts.split('.')[0] if end_ts else ''
            # Build display label: "<event> (HH:MM:SS - HH:MM:SS)"
            time_segment = f" ({start_fmt} - {end_fmt})" if start_fmt and end_fmt else ''
            display_label = f"{action_label}{time_segment}"

            # Update for gr.Video: set value, label (score), and visibility
            video_label = f"Score: {score:.0f}"
            updates.append(gr.update(value=video_url, label=video_label, visible=True))

            # Update for gr.Checkbox: set label with event + time, value (checked), and visibility
            checked = clip_id in selection
            updates.append(gr.update(label=display_label, value=checked, visible=True))
            clip_ids_batch.append(clip_id)
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update(visible=False))
            clip_ids_batch.append("")
    # append clip_ids for State components
    updates.extend(clip_ids_batch)
    # selection controls visibility
    updates.append(gr.update(visible=bool(clips) and status_code=="completed"))
    timer_active = status_code not in ("completed", "failed")
    updates.append(gr.update(active=timer_active))
    # Buttons after completed
    updates.append(gr.update(interactive=bool(selection)))  # create button
    # keep download disabled until file created
    updates.append(gr.update(interactive=False, value=None))
    return updates

def select_all_ui(*args):
    """Select all visible clips in UI and backend"""
    # Last arg is job_id
    job_id = args[-1]
    clip_ids = [cid for cid in args[:-1] if cid]
    # Update backend
    try:
        _ = sw.select_all_clips_sync(job_id)
    except Exception:
        pass
    # Build updates: selected_state, selection_display, checkbox values, download button
        checkbox_updates = [gr.update(value=True) if cid else gr.update(value=False) for cid in args[:-1]]
    create_btn_update = gr.update(interactive=bool(clip_ids))
    download_btn_update = gr.update(interactive=False, value=None)
    return [clip_ids, f"Đã chọn: {clip_ids}", *checkbox_updates, create_btn_update, download_btn_update]

def deselect_all_ui(*args):
    """Deselect all clips in UI and backend"""
    job_id = args[-1]
    try:
        _ = sw.deselect_all_clips_sync(job_id)
    except Exception:
        pass
        checkbox_updates = [gr.update(value=False) for _ in args[:-1]]
    create_btn_update = gr.update(interactive=False)
    download_btn_update = gr.update(interactive=False, value=None)
    return [[], "Đã chọn: []", *checkbox_updates, create_btn_update, download_btn_update]

def create_metadata(job_id, fmt):
    """Trigger metadata creation and return update for download button"""
    return sw.download_selected_clips_sync(job_id, fmt)

def toggle_upload_method(choice):
    """Toggle between file upload and path input"""
    if choice == "Upload File":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

