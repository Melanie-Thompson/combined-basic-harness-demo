"""HF Spaces entry point.

Serves the full custom frontend (chat, logs, Redis, eval panels) via
FastAPI, with a minimal Gradio block mounted at /gradio to satisfy
HF Spaces SDK detection and the ZeroGPU startup check.
"""

import os
import spaces
import gradio as gr

PORT = int(os.getenv("PORT", "7860"))

from main import app as fastapi_app

@spaces.GPU(duration=60)
def _gpu_noop():
    return "ok"

with gr.Blocks() as demo:
    gr.Markdown("Gradio mount — the full UI is at [/](/)")

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT)
