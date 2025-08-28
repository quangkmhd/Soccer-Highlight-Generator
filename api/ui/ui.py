"""Gradio UI construction for the Soccer Action Spotting demo."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import gradio as gr

from .client import SoccerAPIClient

logger = logging.getLogger(__name__)


def create_demo(api_client: SoccerAPIClient) -> gr.Blocks:
    """Create and return the Gradio Blocks app."""
    css = """
    .selected-clip {
        border: 3px solid #22c55e !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(34, 197, 94, 0.5) !important;
    }
    .selected-count {
        background: linear-gradient(45deg, #22c55e, #16a34a);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }
    """

    with gr.Blocks(title="Soccer Action Spotting Demo", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("# Soccer Action Spotting Demo")
        gr.Markdown("Upload a soccer video or specify a local path to generate action clips")

        # States
        selected_clips: gr.State = gr.State(set())
        all_clips_data: gr.State = gr.State([])
        current_viewing_index: gr.State = gr.State(-1)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Video")
                input_type = gr.Radio(
                    choices=["Upload File", "Local Path"], value="Upload File", label="Input Method"
                )
                video_file = gr.File(
                    label="Upload Video File", file_types=[".mp4", ".avi", ".mov", ".mkv"], visible=True
                )
                video_path = gr.Textbox(label="Local Video Path", placeholder="/path/to/your/video.mp4", visible=False)
                process_btn = gr.Button("Start Processing", variant="primary", size="lg")

                gr.Markdown("### Processing Status")
                status_text = gr.Textbox(label="Status", interactive=False)
                progress_text = gr.Textbox(label="Progress", interactive=False, visible=False)
                job_id_display = gr.Textbox(label="Job ID", interactive=False, visible=False)

            with gr.Column(scale=2):
                gr.Markdown("### Generated Action Clips")
                clips_gallery = gr.Gallery(
                    label="Action Clips (Click to preview)",
                    show_label=True,
                    elem_id="clips_gallery",
                    columns=5,
                    rows=4,
                    height="auto",
                    interactive=True,
                    allow_preview=True,
                )

                with gr.Row():
                    select_current_btn = gr.Button("Select Current Clip", variant="primary", size="lg")

                clips_count = gr.Textbox(label="Total Clips", interactive=False, visible=False)

                gr.Markdown("### Selected Clips")
                selected_count = gr.HTML("<div class='selected-count'>0 clips selected</div>")
                selected_clips_gallery = gr.Gallery(
                    label="Selected Clips",
                    show_label=True,
                    elem_id="selected_clips_gallery",
                    columns=3,
                    rows=3,
                    height="300px",
                    interactive=False,
                    visible=True,
                )




        # Handlers
        def toggle_input_visibility(input_method: str) -> Tuple[gr.update, gr.update]:
            if input_method == "Upload File":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        def start_processing(input_method: str, video_file: Optional[str], video_path: Optional[str], prev_job_id: str) -> Tuple[str, str, bool]:
            # Start new job
            if input_method == "Upload File":
                success, result = api_client.upload_video(video_file, None)
            else:
                success, result = api_client.upload_video(None, video_path)
            return ("Processing started...", result, True) if success else (f"Error: {result}", "", False)

        def monitor_progress(job_id: str, current_clips_data: List[Dict[str, Any]]) -> Tuple[str, str, List[Tuple[str, str]], str, bool, List[Dict[str, Any]]]:
            if not job_id:
                return "No job ID", "", [], "", False, []
            status, progress, message, count = api_client.get_status(job_id)
            progress_percent = f"{progress*100:.1f}%" if progress > 0 else ""
            if status == "completed":
                clips_with_labels = api_client.get_clips_gallery(job_id)
                clips_data = api_client.get_clips_data(job_id)
                return f"✅ {message}", "100%", clips_with_labels, f"{count} clips generated", True, clips_data
            if status == "failed":
                return f"❌ Processing failed: {message}", "", [], "", True, []
            return f"🔄 {message}", progress_percent, [], "", False, current_clips_data

        def handle_clip_view(selected_index: gr.SelectData, clips_data: List[Dict[str, Any]], current_selected: Set[str]) -> Tuple[int, str]:
            if not clips_data or selected_index.index >= len(clips_data):
                return -1, "Select Current Clip"
            clip = clips_data[selected_index.index]
            return selected_index.index, ("Unselect Current Clip" if clip["url"] in current_selected else "Select Current Clip")

        def build_selected_gallery(selected: Set[str], clips_data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
            return [
                (clip_data["url"], clip_data["label"]) for clip_data in clips_data if clip_data["url"] in selected
            ]

        def select_current_clip(viewing_index: int, current_selected: Set[str], clips_data: List[Dict[str, Any]]) -> Tuple[Set[str], List[Tuple[str, str]], str, str]:
            if viewing_index < 0 or viewing_index >= len(clips_data):
                return current_selected, build_selected_gallery(current_selected, clips_data), f"<div class='selected-count'>{len(current_selected)} clips selected</div>", "Select Current Clip"
            clip = clips_data[viewing_index]
            clip_url = clip["url"]
            new_selected = set(current_selected)
            if clip_url in new_selected:
                new_selected.remove(clip_url)
                new_button_text = "Select Current Clip"
            else:
                new_selected.add(clip_url)
                new_button_text = "Unselect Current Clip"
            selected_gallery = build_selected_gallery(new_selected, clips_data)
            count_text = f"<div class='selected-count'>{len(new_selected)} clips selected</div>"
            return new_selected, selected_gallery, count_text, new_button_text



        # Wire events
        input_type.change(toggle_input_visibility, inputs=[input_type], outputs=[video_file, video_path])

        process_btn.click(
            start_processing,
            inputs=[input_type, video_file, video_path, job_id_display],
            outputs=[status_text, job_id_display],
        ).then(lambda: gr.update(visible=True), outputs=[progress_text]).then(
            lambda: (set(), [], "Select Current Clip"), outputs=[selected_clips, all_clips_data, select_current_btn]
        )


        clips_gallery.select(
            handle_clip_view, inputs=[all_clips_data, selected_clips], outputs=[current_viewing_index, select_current_btn]
        )

        select_current_btn.click(
            select_current_clip,
            inputs=[current_viewing_index, selected_clips, all_clips_data],
            outputs=[selected_clips, selected_clips_gallery, selected_count, select_current_btn],
        )


        timer = gr.Timer(2.0)
        timer.tick(
            monitor_progress,
            inputs=[job_id_display, all_clips_data],
            outputs=[status_text, progress_text, clips_gallery, clips_count, all_clips_data],
            show_progress=False,
        )

    return demo
