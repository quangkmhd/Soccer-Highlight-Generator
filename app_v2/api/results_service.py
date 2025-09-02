"""
Service for handling results, clip selection, and downloads
"""
import json
import zipfile
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

from .models import HighlightClip

logger = logging.getLogger(__name__)

class ResultsService:
    def __init__(self):
        self.clip_selections: Dict[str, List[str]] = {}  # job_id -> list of selected clip_ids
    
    async def get_job_results(self, job_id: str, result_path: Optional[str]) -> List[HighlightClip]:
        """
        Parse job results from result_path and return highlight clips
        Extract timestamps from video clip filenames with format: Score_start_end_event.mp4
        """
        try:
            # result_path is a directory containing the AI pipeline output
            clips_dir = None
            if os.path.isdir(result_path):
                # Look for subdirectory containing video clips
                for item in os.listdir(result_path):
                    item_path = os.path.join(result_path, item)
                    if os.path.isdir(item_path):
                        clips_dir = item_path
                        break
                
                # If no subdirectory, use the result_path itself
                if not clips_dir:
                    clips_dir = result_path
            else:
                clips_dir = os.path.dirname(result_path)
            
            if not clips_dir or not os.path.exists(clips_dir):
                logger.warning(f"Clips directory not found for job {job_id}: {clips_dir}")
                return []
            
            # Find all MP4 files in the clips directory
            mp4_files = []
            for file in os.listdir(clips_dir):
                if file.endswith('.mp4'):
                    mp4_files.append(file)
            
            if not mp4_files:
                logger.warning(f"No MP4 files found in clips directory for job {job_id}: {clips_dir}")
                return []
            
            clips = []
            for i, filename in enumerate(mp4_files, 1):
                try:
                    # Parse filename format: Score_start_end_event.mp4
                    # Example: 100_01-23-37_01-25-08_Penalty -> Goal.mp4
                    parsed = self._parse_clip_filename(filename)
                    if parsed:
                        # Use actual filename (without extension) as clip_id
                        clip_id = filename.replace('.mp4', '')
                        # Build absolute filesystem path for clip preview
                        abs_path = os.path.abspath(os.path.join(clips_dir, filename))
                        # Convert to relative path within CWD so Gradio allows it
                        cwd = os.getcwd()
                        rel_path = os.path.relpath(abs_path, start=cwd)
                        clips.append(HighlightClip(
                            clip_id=clip_id,
                            start=parsed["start"],
                            end=parsed["end"],
                            label=f"Score {parsed['score']} {parsed['event']}",
                            score=parsed["score"],
                            preview_url=rel_path
                        ))
                except Exception as e:
                    logger.warning(f"Failed to parse clip filename {filename}: {e}")
                    continue
            
            # Sắp xếp clips theo score giảm dần (từ 100 xuống)
            clips.sort(key=lambda clip: clip.score, reverse=True)
            
            logger.info(f"Loaded {len(clips)} clips for job {job_id} from {clips_dir}")
            return clips
            
        except Exception as e:
            logger.error(f"Failed to load results for job {job_id}: {e}")
            return []
    
    async def save_clip_selection(self, job_id: str, clip_ids: List[str]) -> bool:
        """
        Save user's clip selection for a job
        """
        try:
            self.clip_selections[job_id] = clip_ids
            logger.info(f"Saved clip selection for job {job_id}: {clip_ids}")
            return True
        except Exception as e:
            logger.error(f"Failed to save clip selection for job {job_id}: {e}")
            return False
    
    async def get_clip_selection(self, job_id: str) -> List[str]:
        """
        Get saved clip selection for a job
        """
        return self.clip_selections.get(job_id, [])
    
    async def create_clips_zip(self, job_id: str, clips: List[HighlightClip], mode: str = "all", output_name: str | None = None) -> str:
        """
        Create ZIP file with selected clips
        Returns path to the created ZIP file
        """
        try:
            # Get clip selection if mode is "selected"
            selected_clip_ids = []
            if mode == "selected":
                selected_clip_ids = await self.get_clip_selection(job_id)
                if not selected_clip_ids:
                    raise Exception("No clips selected for download")
            
            # Create temporary ZIP file
            temp_dir = tempfile.mkdtemp()
            base_name = output_name if output_name else f"highlights_{job_id}"
            zip_path = os.path.join(temp_dir, f"{base_name}.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                for clip in clips:
                    # Skip non-selected clips when mode is "selected"
                    if mode != "all" and clip.clip_id not in selected_clip_ids:
                        continue

                    # Resolve absolute path from preview_url (stored as path relative to CWD)
                    abs_path = os.path.abspath(clip.preview_url)
                    if not os.path.isfile(abs_path):
                        logger.warning(f"Clip file not found: {abs_path}")
                        continue

                    # Add the actual video clip to the ZIP using its filename as the archive name
                    arcname = os.path.basename(abs_path)
                    zip_file.write(abs_path, arcname=arcname)
            
            logger.info(f"Created ZIP file for job {job_id} as {base_name}: {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Failed to create clips ZIP for job {job_id}: {e}")
            raise
    
    async def generate_metadata(self, job_id: str, clips: List[HighlightClip], format_type: str, mode: str = "all", output_name: str | None = None) -> str:
        """
        Generate metadata file in SRT or XML format
        Returns path to the generated metadata file
        """
        try:
            # Filter clips based on mode
            if mode == "selected":
                selected_clip_ids = await self.get_clip_selection(job_id)
                if not selected_clip_ids:
                    raise Exception("No clips selected for metadata export")
                # Filter clips to only include selected ones
                filtered_clips = [clip for clip in clips if clip.clip_id in selected_clip_ids]
            else:
                # Use all clips
                filtered_clips = clips
            
            if not filtered_clips:
                raise Exception("No clips available for metadata export")
            
            temp_dir = tempfile.mkdtemp()
            # Determine base filename
            base_name = output_name if output_name else f"highlights_{job_id}"
            
            if format_type.lower() == "srt":
                metadata_path = os.path.join(temp_dir, f"{base_name}.srt")
                content = self._generate_srt_content(filtered_clips)
            elif format_type.lower() == "xml":
                metadata_path = os.path.join(temp_dir, f"{base_name}.xml")  
                content = self._generate_xml_content(filtered_clips)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated {format_type} metadata for job {job_id} ({mode} mode) as {base_name}: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Failed to generate {format_type} metadata for job {job_id}: {e}")
            raise
    
    def _seconds_to_timestamp(self, seconds) -> str:
        """Convert seconds to HH:MM:SS.mmm format"""
        try:
            # Handle string or numeric input
            if isinstance(seconds, str):
                seconds = float(seconds)
            elif seconds is None:
                seconds = 0.0
            
            seconds = float(seconds)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            milliseconds = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert timestamp: {seconds}, error: {e}")
            return "00:00:00.000"
    
    def _parse_clip_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse clip filename with format: Score_start_end_event.mp4
        Example: 100_01-23-37_01-25-08_Penalty->Goal.mp4
        Returns dict with score, start, end, event
        """
        try:
            # Remove .mp4 extension
            name = filename.replace('.mp4', '')
            
            # Split by underscores
            parts = name.split('_')
            if len(parts) < 4:
                logger.warning(f"Invalid filename format: {filename}")
                return None
            
            # Extract parts
            score = int(parts[0])
            start_time = parts[1]  # Format: HH-MM-SS
            end_time = parts[2]    # Format: HH-MM-SS
            event = '_'.join(parts[3:])  # Join remaining parts as event name
            
            # Convert time format from HH-MM-SS to HH:MM:SS.000
            start_timestamp = self._convert_time_format(start_time)
            end_timestamp = self._convert_time_format(end_time)
            
            return {
                'score': score,
                'start': start_timestamp,
                'end': end_timestamp,
                'event': event
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse filename {filename}: {e}")
            return None
    
    def _convert_time_format(self, time_str: str) -> str:
        """
        Convert time format from HH-MM-SS to HH:MM:SS.000
        """
        try:
            # Replace hyphens with colons and add milliseconds
            converted = time_str.replace('-', ':') + '.000'
            return converted
        except Exception as e:
            logger.warning(f"Failed to convert time format {time_str}: {e}")
            return "00:00:00.000"
    
    def _generate_srt_content(self, clips: List[HighlightClip]) -> str:
        """Generate SRT subtitle format content"""
        srt_content = ""
        for i, clip in enumerate(clips, 1):
            # Convert timestamp format from HH:MM:SS.mmm to HH:MM:SS,mmm (SRT uses comma)
            start_srt = clip.start.replace('.', ',')
            end_srt = clip.end.replace('.', ',')
            
            srt_content += f"""{i}
{start_srt} --> {end_srt}
{clip.label}

"""
        return srt_content.strip()
    
    def _generate_xml_content(self, clips: List[HighlightClip]) -> str:
        """Generate XML format content"""
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<highlights>\n'
        
        for clip in clips:
            xml_content += f"""  <clip id="{clip.clip_id}">
<start>{clip.start}</start>
<end>{clip.end}</end>
<label>{clip.label}</label>
<score>{clip.score}</score>
</clip>
"""
        
        xml_content += '</highlights>'
        return xml_content

# Global results service instance
results_service = ResultsService()
