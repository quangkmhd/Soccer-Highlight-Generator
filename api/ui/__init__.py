"""Gradio demo package for Soccer Action Spotting.
Exposes client and UI constructors for reuse.
"""
from .client import SoccerAPIClient
from .ui import create_demo

__all__ = ["SoccerAPIClient", "create_demo"]
