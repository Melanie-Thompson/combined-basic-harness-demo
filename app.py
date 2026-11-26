"""HF Spaces entry point (Gradio SDK).

Launches the FastAPI harness with Gradio mounted on top.
HF Spaces expects `demo.launch()` or a FastAPI `app` — we provide both.
"""

import os
import gradio as gr
import spaces

# Set port for HF Spaces
PORT = int(os.getenv("PORT", "7860"))

# Import the FastAPI app (triggers lifespan, loads tools)
from main import app as fastapi_app

# Simple Gradio landing that links to the full UI
with gr.Blocks(title="Agent Harness Demo") as demo:
    gr.Markdown(
        """
        # Agent Harness Demo

        Plan-and-execute LangGraph agent with:
        - Tool verification and cryptographic signing
        - Faithfulness checking against tool outputs
        - Live evaluation metrics
        - TDA topology visualization

        **The full UI is served at the root path** — the chat interface, log panel,
        Redis inspector, and eval dashboard are all part of the custom frontend.
        """
    )
    with gr.Row():
        prompt = gr.Textbox(label="Message", placeholder="e.g. What's the weather in London?", scale=4)
        send = gr.Button("Send", variant="primary", scale=1)

    output = gr.Textbox(label="Response", lines=8, interactive=False)
    tool_log = gr.Textbox(label="Tool Calls", lines=4, interactive=False)

    @spaces.GPU(duration=60)
    def _gpu_noop():
        pass

    async def chat(message):
        import traceback
        from AI.graph import stream_reply
        from main import _registry

        try:
            tokens = []
            tools = []
            blocked = ""

            async for evt in stream_reply(
                message,
                registry=_registry,
                session_id="gradio_demo",
            ):
                evt_type = evt.get("type", "")
                if evt_type == "token":
                    tokens.append(evt["token"])
                elif evt_type == "tool_call":
                    tools.append(f"-> {evt['tool']}({evt.get('args', {})})")
                elif evt_type == "tool_result":
                    tools.append(f"<- {evt['tool']}: {evt.get('result', '')[:200]}")
                elif evt_type == "tool_blocked":
                    blocked = evt.get("reason", "Blocked")

            response = blocked if blocked else "".join(tokens)
            tool_text = "\n".join(tools) if tools else "(no tool calls)"
            return response, tool_text
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error: {e}\n\n{tb}", "(error)"

    send.click(fn=chat, inputs=prompt, outputs=[output, tool_log])
    prompt.submit(fn=chat, inputs=prompt, outputs=[output, tool_log])

# For HF Spaces Gradio SDK
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT)
