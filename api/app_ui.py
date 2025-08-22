"""Application entrypoint for running the Gradio demo."""
from __future__ import annotations

import logging

from ui.client import SoccerAPIClient
from ui.ui import create_demo


def main() -> None:
    """Run the Gradio demo locally."""
    logging.basicConfig(level=logging.INFO)
    api_client = SoccerAPIClient()
    demo = create_demo(api_client)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)


if __name__ == "__main__":
    main()
