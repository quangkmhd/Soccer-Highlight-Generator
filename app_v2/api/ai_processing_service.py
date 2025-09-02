"""
AI Processing Service - Core AI inference functionality
This is the most important service that handles actual AI video processing
"""
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

# Add root directory to path for inference imports
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# Import main AI processing function - this is the core functionality  
from inference.parallel_inference import run_inference_pipeline_parallel

logger = logging.getLogger(__name__)

class AIProcessingService:
    """
    Core AI Processing Service
    Handles the actual AI inference pipeline execution
    """
    
    @staticmethod
    async def process_video(video_path: str, job_id: str = None, video_id: str = None) -> Dict[str, Any]:
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting AI processing - Job: {job_id}, Video: {video_id}")
            
            video_path_obj = Path(video_path)
            config_path = Path("inference/inference_config.yaml")
            
            # Execute AI inference pipeline in executor to avoid blocking
            output_dir = await asyncio.get_event_loop().run_in_executor(
                None, 
                run_inference_pipeline_parallel,
                video_path_obj,
                config_path
            )
            
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            
            result = {
                "success": True,
                "output_dir": str(output_dir),
                "processing_duration": processing_duration,
                "started_at": start_time.isoformat(),
                "completed_at": end_time.isoformat(),
                "video_path": str(video_path),
                "message": "AI processing completed successfully"
            }
            
            logger.info(f"AI processing completed - Job: {job_id}, Duration: {processing_duration:.2f}s, Output: {output_dir}")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            processing_duration = (end_time - start_time).total_seconds()
            
            error_result = {
                "success": False,
                "error": str(e),
                "processing_duration": processing_duration,
                "started_at": start_time.isoformat(),
                "failed_at": end_time.isoformat(),
                "video_path": str(video_path),
                "message": f"AI processing failed: {str(e)}"
            }
            
            logger.error(f"AI processing failed - Job: {job_id}, Duration: {processing_duration:.2f}s, Error: {e}")
            raise Exception(f"AI processing failed: {str(e)}")

# Global service instance
ai_processing_service = AIProcessingService()
