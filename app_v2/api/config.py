"""
Configuration management using YAML
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any

# Global config cache
_config_cache: Dict[str, Any] = None

def _ensure_directories(config: Dict[str, Any]) -> None:
    """Ensure required directories exist"""
    app_root = Path(__file__).resolve().parent.parent
    
    # Video upload directory
    upload_dir = app_root / config['video']['upload_dir']
    upload_dir.mkdir(parents=True, exist_ok=True)

def get_config() -> Dict[str, Any]:
    """Load and cache configuration from YAML file"""
    global _config_cache
    
    if _config_cache is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f)
        
        # Ensure required directories exist after loading config
        if _config_cache:
            _ensure_directories(_config_cache)
    
    return _config_cache

# Initialize config
config = get_config()

def get_upload_dir() -> Path:
    """Get upload directory path"""
    return Path(__file__).resolve().parent.parent / config['video']['upload_dir']

def get_api_base_url() -> str:
    """Get API base URL"""
    return config.get('api', {}).get('base_url', 'http://localhost:8000/api/v2')

def get_gradio_config() -> Dict[str, Any]:
    """Get Gradio configuration"""
    return config.get('gradio', {
        'server_name': '0.0.0.0',
        'server_port': 7860,
        'debug': True,
        'share': False
    })

def get_supported_video_formats() -> list:
    """Get supported video formats with dot prefix"""
    formats = config.get('video', {}).get('supported_formats', ['mp4'])
    return [f".{fmt}" if not fmt.startswith('.') else fmt for fmt in formats]
