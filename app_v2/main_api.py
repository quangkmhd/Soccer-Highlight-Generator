"""
FastAPI Application Entry Point
"""
import asyncio
import logging
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Ensure project root is on sys.path when running this file directly
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_v2.api.routes import video_router
from app_v2.api.job_manager import job_manager
from app_v2.api.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting Soccer Highlight Detection API")
    
    # Start job processor
    await job_manager.start_processor()
    logger.info("Job processor started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Soccer Highlight Detection API")
    
    # Stop job processor
    await job_manager.stop_processor()
    logger.info("Job processor stopped")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Soccer Highlight Detection API",
    description="AI-powered soccer highlight detection with queue management",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video_router, prefix="/api/v2", tags=["videos"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Soccer Highlight Detection API v2.0", 
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    queue_info = await job_manager.get_queue_info()
    
    return {
        "status": "healthy",
        "queue": {
            "length": queue_info["queue_length"],
            "processing": queue_info["processing"],
            "current_job": queue_info["current_job"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    
    uvicorn.run(
        "app_v2.main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
