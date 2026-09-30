import sys
import os
# Ensure the root directory is in the path so av_pipeline imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import cv2
import json
import time
import gradio as gr
import numpy as np
from PIL import ImageGrab
from gradio_client import Client, handle_file

# Inject a pre-compiled FFmpeg binary into the Windows PATH so Gradio can process browser audio
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
except ImportError:
    print("Warning: imageio_ffmpeg not installed. Gradio may fail to process mic audio.")

# Default URL, user can override it in the UI
DEFAULT_KAGGLE_URL = "https://<your-kaggle-url>.gradio.live"

def execute_payload(payload_json):
    import pyautogui
    try:
        if isinstance(payload_json, str):
            data = json.loads(payload_json)
        else:
            data = payload_json
            
        metadata = data.get("metadata", {})
        norm_pos = metadata.get("norm_pos", [0.0, 0.0])
        action = metadata.get("action", "")
        
        nx, ny = norm_pos[0], norm_pos[1]
        if nx != 0.0 and ny != 0.0:
            sw, sh = pyautogui.size()
            abs_x = int(nx * sw)
            abs_y = int(ny * sh)
            print(f"[Actuator] Executing {action} at ({abs_x}, {abs_y})")
            pyautogui.moveTo(abs_x, abs_y, duration=0.2)
            if action in ["click", "launch_app"]:
                pyautogui.click()
                
            # If they want to type
            if action == "type_text" and "text" in metadata:
                text_to_type = metadata["text"]
                pyautogui.write(text_to_type, interval=0.05)
                
            return f"Executed '{action}' at screen coordinates ({abs_x}, {abs_y})"
        return "No specific UI element found to click."
    except Exception as e:
        return f"Error executing payload: {e}"

def process_command(audio_filepath, kaggle_url):
    if not audio_filepath:
        return "No audio provided."
    if "<your-kaggle-url>" in kaggle_url:
        return "Please paste your active Kaggle Gradio URL above!"
        
    print("Taking screenshot of local machine...")
    pil_img = ImageGrab.grab()
    screen_path = "temp_screen.jpg"
    pil_img.save(screen_path)
    
    print(f"Sending to Kaggle GPU: {kaggle_url}")
    try:
        client = Client(kaggle_url)
        t0 = time.time()
        result = client.predict(
            audio_path=handle_file(audio_filepath),
            image_path=handle_file(screen_path),
            api_name="/evaluate"
        )
        decision_json, payload_json = result
        latency = time.time() - t0
        
        dec = decision_json if isinstance(decision_json, dict) else json.loads(decision_json)
        msg = f"Kaggle GPU Latency: {latency:.2f} seconds\nAction: {dec.get('action')} on {dec.get('target')}"
        
        exec_msg = execute_payload(payload_json)
        return f"{msg}\nActuator: {exec_msg}"
    except Exception as e:
        return f"Error connecting to Kaggle: {e}"

with gr.Blocks(title="JEV Thin Client", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ JEV AI Web Client")
    gr.Markdown("Run this locally! It records your voice securely in the browser, grabs your physical screen, and sends both to your Kaggle GPU for lightning-fast processing.")
    
    with gr.Row():
        url_input = gr.Textbox(label="Kaggle Server URL (Update this when you restart Kaggle)", value=DEFAULT_KAGGLE_URL)
        
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Voice Command")
        
    btn = gr.Button("Send Command to AI", variant="primary")
    out = gr.Textbox(label="Result Logs", lines=4)
    
    btn.click(process_command, inputs=[audio_in, url_input], outputs=[out])

if __name__ == "__main__":
    print("Starting Local Web Client...")
    app.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
