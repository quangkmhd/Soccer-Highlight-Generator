"""
Soccer Highlight Detection - Gradio Web Interface (Refactored)
"""
import sys
from pathlib import Path

# Add project root to sys.path to allow for absolute imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app_v2.ui import create_gradio_interface
from app_v2.api.config import get_gradio_config

# Create and configure demo interface
demo = create_gradio_interface()

if __name__ == "__main__":
    config = get_gradio_config()
    demo.launch(
        server_name=config.get('server_name', '0.0.0.0'),
        server_port=config.get('server_port', 7860),
        share=config.get('share'),
        debug=config.get('debug')
    )
