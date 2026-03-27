"""app.py — Hugging Face Space entry point (under 50 lines).

This file only imports and launches the Gradio app.
All logic lives in src/handlers.py.

To run locally:
    python app.py

The Space runs this file automatically on startup.
"""

import logging

from src.handlers import build_demo

# Set up basic logging so errors appear in the Space logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)

# Build the Gradio app (all UI and event wiring defined in handlers.py)
demo = build_demo()

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
