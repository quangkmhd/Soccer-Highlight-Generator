
import asyncio
import logging
import os
import shutil
import uuid
import queue
import logging.handlers
import sys
import re
from contextlib import asynccontextmanager
from pathlib import Path

# --- Direct import from pipeline script ---
# Ensure the project root is in the Python path to allow the import
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.append(str(PIPELINE_ROOT))
    
from run_pipeline import main as run_pipeline_main

import yaml
from fastapi import (FastAPI, File, Form, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketState

# --- Constants & Global State ---
SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DIR = SCRIPT_DIR / "static"
TEMP_DIR_NAME = "temp_processing"
TEMP_DIR = SCRIPT_DIR / TEMP_DIR_NAME
CLIPS_DIR_NAME = "final_clips"
CLIPS_DIR = SCRIPT_DIR / CLIPS_DIR_NAME

# Ensure required directories exist
TEMP_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] [%(name)-15s] %(message)s")
logger = logging.getLogger("Backend")

# --- Custom Logging Handler ---
class QueueLogHandler(logging.Handler):
    """A logging handler that puts records into a queue."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        # Put the formatted message directly into the queue
        self.log_queue.put(self.format(record))

# --- Helper Functions ---
def get_config():
    """Loads the main pipeline configuration."""
    config_path = PIPELINE_ROOT / "config.yml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config.yml: {e}")
        raise HTTPException(status_code=500, detail="Could not load server configuration.")

def get_start_time_from_filename(filename: str) -> int:
    """
    Extracts the start time in seconds from a clip filename.
    Filename format is expected to be 'H-MM-SS--...mp4'.
    Returns a large number if the format is not found, to sort these last.
    """
    match = re.match(r'(\d+)-(\d{2})-(\d{2})--', filename)
    if match:
        h, m, s = map(int, match.groups())
        return h * 3600 + m * 60 + s
    return 999999 # A large number for files that don't match the pattern

def get_video_output_dir(video_name: str) -> Path:
    """Gets the specific output directory for clips for a given video."""
    config = get_config()
    base_clips_dir_name = Path(config['output_dirs']['clips_dir']).name
    video_clips_dir = STATIC_DIR / base_clips_dir_name / video_name
    video_clips_dir.mkdir(parents=True, exist_ok=True)
    return video_clips_dir

async def run_pipeline_async(video_path: str, video_name: str, websocket: WebSocket):
    """
    Runs the main pipeline script as an imported module and streams logs
    to the client via a WebSocket.
    """
    log_queue = queue.Queue()
    
    # Configure a custom handler to capture logs from the pipeline module
    formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] [%(name)-15s] %(message)s")
    queue_handler = QueueLogHandler(log_queue)
    queue_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    is_running = True

    async def log_forwarder():
        """Pulls logs from the sync queue and sends them over the async websocket."""
        while is_running or not log_queue.empty():
            try:
                log_message = await asyncio.to_thread(log_queue.get, timeout=0.1)
                
                stage_keywords = {
                    "STAGE 1": {"num": 1, "name": "Converting video to 25 FPS", "progress": 15},
                    "STAGE 2": {"num": 2, "name": "Ball-action prediction", "progress": 30},
                    "STAGE 3": {"num": 3, "name": "Camera segmentation", "progress": 50},
                    "STAGE 4": {"num": 4, "name": "Processing and merging predictions", "progress": 70},
                    "STAGE 5": {"num": 5, "name": "Applying rules to generate highlights", "progress": 85},
                    "STAGE 6": {"num": 6, "name": "Cutting highlight clips", "progress": 95},
                }
                stage_updated = False
                for keyword, data in stage_keywords.items():
                    if keyword in log_message:
                        await websocket.send_json({
                            "status": "progress", "log": log_message,
                            "stage": data["num"], "stage_name": data["name"], "progress": data["progress"]
                        })
                        stage_updated = True
                        break
                if not stage_updated:
                    await websocket.send_json({"status": "progress", "log": log_message})
            except queue.Empty:
                await asyncio.sleep(0.1)
    
    forwarder = asyncio.create_task(log_forwarder())

    try:
        await websocket.send_json({"status": "starting", "message": "Pipeline initiated..."})
        
        config_path = PIPELINE_ROOT / "config.yml"
        video_path_obj = Path(video_path)
        await asyncio.to_thread(run_pipeline_main, video_path_obj, config_path)

        await websocket.send_json({"status": "clips_ready", "message": "Pipeline completed. Fetching clips..."})

        # FIX: Determine the original video stem to find the correct output directory
        temp_video_stem = Path(video_name).stem
        parts = temp_video_stem.split('_')
        original_video_stem = temp_video_stem
        # The web UI adds a short UUID like '_xxxxxx'. We look for it and remove it.
        if len(parts) > 1 and len(parts[-1]) == 8:
            try:
                # Check if the last part is a hex string (like a UUID snippet)
                int(parts[-1], 16)
                original_video_stem = '_'.join(parts[:-1])
            except ValueError:
                # Not a hex string, so it's part of the original name
                pass
        
        # Use the corrected original_video_stem to find the clips (search recursively)
        output_dir = get_video_output_dir(original_video_stem)
        base_clips_dir = STATIC_DIR / Path(get_config()['output_dirs']['clips_dir']).name
        
        # Gather all clip files and sort them by the timestamp in their name
        all_clip_files = list(output_dir.rglob("*.mp4"))
        all_clip_files.sort(key=lambda f: get_start_time_from_filename(f.name))

        clips = []
        for clip_file in all_clip_files:
            try:
                rel_path = clip_file.relative_to(base_clips_dir)
                clips.append(f"/clips/{rel_path.as_posix()}")
            except ValueError:
                # Should not happen, but ignore if the path is outside base dir
                continue
        
        logger.info(f"Found and sorted {len(clips)} clips in '{output_dir.relative_to(PIPELINE_ROOT)}' to send to frontend.")

        await websocket.send_json({"status": "completed", "clips": clips, "progress": 100})

    except asyncio.CancelledError:
        logger.warning(f"Pipeline task for {video_name} was cancelled by websocket disconnect.")
        logger.warning("NOTE: The underlying pipeline thread will continue to run until it completes, as it cannot be safely terminated.")
        raise
    except Exception as e:
        error_msg = f"An unexpected error occurred in the pipeline: {e}"
        logger.error(error_msg, exc_info=True)
        await websocket.send_json({"status": "progress", "log": f"❌ ERROR: {error_msg}"})
        await websocket.send_json({"status": "error", "message": error_msg})
    finally:
        is_running = False
        await forwarder
        root_logger.removeHandler(queue_handler)
        root_logger.setLevel(original_level)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    logger.info("--- Backend Server Starting Up ---")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(exist_ok=True)
    yield
    logger.info("--- Backend Server Shutting Down ---")

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FastAPI Routes ---
@app.post("/api/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """Handles video file upload."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
        
    # Generate a unique-ish name to avoid collisions
    unique_id = uuid.uuid4().hex[:8]
    video_name = f"{Path(video.filename).stem}_{unique_id}.mp4"
    file_path = TEMP_DIR / video_name

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        logger.info(f"Successfully saved video to {file_path}")
        return JSONResponse(
            content={"message": "Video uploaded successfully", "video_name": video_name},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Could not save video: {e}")
        raise HTTPException(status_code=500, detail=f"Could not save video: {e}")

@app.websocket("/ws/run-inference/{video_name}")
async def run_inference_ws(websocket: WebSocket, video_name: str):
    """WebSocket endpoint to run inference and stream logs."""
    await websocket.accept()
    video_path = TEMP_DIR / video_name
    
    if not video_path.exists():
        await websocket.send_json({"status": "error", "message": f"Video {video_name} not found."})
        await websocket.close()
        return

    task = asyncio.create_task(
        run_pipeline_async(str(video_path), video_name, websocket)
    )

    try:
        await task
    except WebSocketDisconnect:
        logger.warning(f"WebSocket disconnected for {video_name}. Cancelling pipeline task.")
        task.cancel()
    except Exception as e:
        logger.error(f"WebSocket error for {video_name}: {e}")
        task.cancel()
    finally:
        # Check if the websocket is still open before trying to close it
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
                logger.info(f"WebSocket for {video_name} closed gracefully.")
            except RuntimeError as e:
                # This can happen if the client disconnects abruptly
                logger.warning(f"Could not close WebSocket for {video_name}, it was likely already closed: {e}")
        else:
            logger.info(f"WebSocket for {video_name} was already closed.")


@app.post("/api/merge-clips")
async def merge_clips(clips: list[str] = Form(...)):
    """Merges selected video clips into a single highlight reel."""
    if not clips:
        raise HTTPException(status_code=400, detail="No clips provided to merge.")

    cfg = get_config()
    base_clips_dir_name = Path(cfg['output_dirs']['clips_dir']).name
    base_clips_dir = STATIC_DIR / base_clips_dir_name

    clip_paths = []
    for url in clips:
        if not url.startswith("/clips/"):
            raise HTTPException(status_code=400, detail=f"Invalid clip URL: {url}")
        relative_part = url[len("/clips/"):]  # Strip the '/clips/' prefix
        file_path = base_clips_dir / Path(relative_part)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Clip not found: {file_path.name}")
        clip_paths.append(file_path)

    # Verify at least one path
    if not clip_paths:
        raise HTTPException(status_code=404, detail="No valid clips found to merge.")

    merged_filename = f"highlight_reel_{uuid.uuid4().hex[:8]}.mp4"
    output_path = CLIPS_DIR / merged_filename
    file_list_path = TEMP_DIR / f"merge_list_{uuid.uuid4().hex}.txt"

    try:
        with open(file_list_path, 'w', encoding='utf-8') as f:
            for path in clip_paths:
                f.write(f"file '{path.resolve()}'\n")

        command = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(file_list_path),
            "-c", "copy",
            str(output_path)
        ]

        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace').strip()
            logger.error(f"FFmpeg failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Failed to merge clips: {error_msg}")

        logger.info(f"Successfully merged {len(clips)} clips into {output_path}")
        return JSONResponse(
            content={
                "message": "Clips merged successfully!",
                "video_url": f"/final_clips/{merged_filename}",
                "download_url": f"/api/download/{merged_filename}"
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error merging clips: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred while merging clips: {e}")
    finally:
        file_list_path.unlink(missing_ok=True)

# New endpoint to force download
@app.get("/api/download/{filename:path}")
async def download_merged_clip(filename: str):
    """Provides a downloadable link for a merged clip."""
    file_path = CLIPS_DIR / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        path=file_path,
        media_type='application/octet-stream',
        filename=filename
    )


# Serve the final merged clips for streaming in the <video> player
app.mount("/final_clips", StaticFiles(directory=CLIPS_DIR), name="final_clips")

# Serve the individual clips generated by the pipeline
# The path needs to match get_video_output_dir structure
config = get_config()
base_clips_dir_name = Path(config['output_dirs']['clips_dir']).name
individual_clips_dir = STATIC_DIR / base_clips_dir_name
# FIX: Ensure the directory for individual clips exists before mounting
individual_clips_dir.mkdir(parents=True, exist_ok=True)
app.mount(f"/clips", StaticFiles(directory=individual_clips_dir), name="clips")


# Serve the React frontend (assuming it's built into a 'build' directory)
frontend_build_dir = PIPELINE_ROOT / "web_ui/frontend/build"
# FIX: Ensure build directory exists to prevent crash on startup, especially in dev
frontend_build_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_build_dir, html=True), name="static_frontend")


@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """Catches all other routes and serves the React index.html."""
    index_path = frontend_build_dir / "index.html"
    if not index_path.exists():
        return JSONResponse(
            status_code=404,
            content={"detail": "Frontend not found. Please run `npm run build` in 'web_ui/frontend' for production use."}
        )
    return FileResponse(index_path)

if __name__ == "__main__":
    import uvicorn
    # Make sure to run from the 'backend' directory for relative paths to work
    os.chdir(SCRIPT_DIR)
    uvicorn.run(app, host="0.0.0.0", port=8000) 