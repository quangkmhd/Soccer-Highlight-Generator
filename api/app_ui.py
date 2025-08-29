"""Application entrypoint for running the Gradio demo."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from ui.client import SoccerAPIClient
from ui.ui import create_demo


def _load_server_config() -> Dict[str, Any]:
    config_path = Path(__file__).parent / "config_api.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_base_url_from_config(cfg: Dict[str, Any]) -> str:
    # Prefer explicit public base URL when deploying for external users
    public_base = cfg.get("public_base_url")
    if isinstance(public_base, str) and public_base.strip():
        return public_base.rstrip("/")

    server = cfg.get("server", {})
    host = str(server.get("host", "localhost"))
    port = int(server.get("port", 8000))
    # When API binds to 0.0.0.0, clients should use localhost
    client_host = "localhost" if host in {"0.0.0.0", "::"} else host
    return f"http://{client_host}:{port}"


def main() -> None:
    """Run the Gradio demo locally."""
    logging.basicConfig(level=logging.INFO)
    cfg = _load_server_config()
    base_url = _build_base_url_from_config(cfg)
    api_client = SoccerAPIClient(base_url=base_url)
    demo = create_demo(api_client)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)


if __name__ == "__main__":
    main()
