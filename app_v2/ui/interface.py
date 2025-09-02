"""
Gradio interface creation and layout
"""
import os
import gradio as gr
from app_v2.ui import ui_handlers as uh
from app_v2.ui.sync_wrappers import start_processing_sync, auto_upload_on_file, auto_register_on_path
from app_v2.api.config import get_supported_video_formats


# --- Auto handlers to hide buttons and trigger upload/register automatically ---
def create_gradio_interface():
    """Create and return the Gradio interface"""
    with gr.Blocks(title="Soccer Highlight Detection", theme=gr.themes.Soft(), head=uh.load_js_validation()) as demo:
        
        gr.Markdown("# ⚽ Soccer Highlight Detection")
        gr.Markdown("Auto create metadata, clips highlights for soccer video")
        
        with gr.Row():
            # Left Column
            with gr.Column(scale=1):
                gr.Markdown("## 📹 Upload Video")
                
                # Upload method selection
                upload_method = gr.Radio(
                    choices=["Upload File", "Input Path"],
                    value="Upload File",
                    label="Choses upload method"
                )
                
                # File upload section
                with gr.Group(visible=True) as file_upload_group:
                    file_input = gr.File(
                        label="Upload file video",
                        file_types=get_supported_video_formats(),
                        type="filepath",
                        elem_id="video-file-input" 
                    )
                
                # Path input section  
                with gr.Group(visible=False) as path_input_group:
                    path_input = gr.Textbox(
                        label="Input path video",
                        placeholder="/path/to/video.mp4"
                    )
                   
                # Status 
                video_id_display = gr.Textbox(label="Video ID", interactive=False, visible=False)
                upload_status = gr.Textbox(label="Upload status", interactive=False)

                process_btn = gr.Button("🎬 Start Processing", variant="primary", size="lg")

                # Job status
                job_id_display = gr.Textbox(label="Job ID", interactive=False, visible=False)
                job_status = gr.Textbox(label="Processing status", interactive=False)

            # Right Column
            with gr.Column(scale=2):
                gr.Markdown("## 🎥 Highlight Clips")

                with gr.Group() as clips_group:
                    # ----- State & Display for selections -----
                    selected_state = gr.State([])
                    selection_display = gr.Markdown("Đã chọn: []", visible=False)

                    # ----- Dynamic clip grid (video + checkbox) -----
                    MAX_CLIPS = 100 
                    video_comps = []
                    checkbox_comps = []
                    clipid_states = []

                    with gr.Row() as clips_row:
                        for _i in range(MAX_CLIPS):
                            with gr.Column() as _col:
                                vid = gr.Video(visible=False, height=190, show_download_button=True)
                                cb  = gr.Checkbox(label="", visible=False)
                            video_comps.append(vid)
                            checkbox_comps.append(cb)
                            clipid_states.append(gr.State(""))

                    # Selection controls
                    with gr.Row(visible=False) as selection_controls:
                        select_all_btn = gr.Button("✅ Select all", variant="secondary")
                        deselect_all_btn = gr.Button("❌ Deselect all", variant="secondary")

                # Download section
                gr.Markdown("## 📥 Download Options")
                # Download controls defined BEFORE we attach events that reference the button
                with gr.Row():
                    download_format = gr.Radio(
                        choices=["SRT", "XML"],
                        value="SRT",
                        label="Định dạng tải xuống"
                    )
                    with gr.Column(scale=1):
                        # Button 1: create metadata file (generate)
                        create_meta_btn = gr.Button(
                            "📝 Create metadata file",
                            variant="secondary",
                            interactive=False  # enable khi đã chọn clip
                        )
                        # Button 2: actual download button, không có callback
                        download_meta_btn = gr.DownloadButton(
                            "📥 Download metadata",
                            variant="primary",
                            interactive=False  # enable + có value sau khi tạo file
                        )

                # Attach click event for download button so it recomputes on every press
                create_meta_btn.click(
                    uh.create_metadata,
                    inputs=[job_id_display, download_format],
                    outputs=[download_meta_btn]
                )

                # Attach checkbox events (placed AFTER button is defined)
                for idx, cb in enumerate(checkbox_comps):
                    cb.change(
                        uh.toggle_clip,
                        inputs=[cb, selected_state, clipid_states[idx], job_id_display],
                        outputs=[selected_state, selection_display, create_meta_btn, download_meta_btn]
                    )
                
        # Auto-refresh timer for automatic results loading
        timer = gr.Timer(2.0, active=False)  # Check every 2 seconds, initially inactive

        # Event handlers
        upload_method.change(
            uh.toggle_upload_method,
            inputs=[upload_method],
            outputs=[file_upload_group, path_input_group]
        )
        # Auto-trigger upload/register and hide buttons when inputs change
        file_input.change(
            auto_upload_on_file,
            inputs=[file_input],
            outputs=[video_id_display, upload_status]
        )

        path_input.change(
            auto_register_on_path,
            inputs=[path_input],
            outputs=[video_id_display, upload_status]
        )
        
        process_btn.click(
            start_processing_sync,
            inputs=[video_id_display],
            outputs=[job_id_display, job_status, timer]
        )
       
        # Build outputs list for timer (after all components created)
        output_components = [job_status, selection_display]
        for vid, cb in zip(video_comps, checkbox_comps):
            output_components.extend([vid, cb])
        output_components.extend(clipid_states)
        output_components.extend([selection_controls, timer, create_meta_btn, download_meta_btn])

        # Configure timer tick behavior -> use refresh()
        timer.tick(
            lambda job_id, sel: uh.refresh(job_id, sel, MAX_CLIPS),
            inputs=[job_id_display, selected_state],
            outputs=output_components,
            show_progress=False
        )
        
        # Selection control handlers (update UI + backend)
        select_all_btn.click(
            uh.select_all_ui,
            inputs=[*clipid_states, job_id_display],
            outputs=[selected_state, selection_display, *checkbox_comps, create_meta_btn, download_meta_btn]
        )

        deselect_all_btn.click(
            uh.deselect_all_ui,
            inputs=[*clipid_states, job_id_display],
            outputs=[selected_state, selection_display, *checkbox_comps, create_meta_btn, download_meta_btn]
        )
        
    
    return demo
