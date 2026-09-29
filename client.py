import os
import cv2
import json
import time
import numpy as np
from PIL import ImageGrab
from gradio_client import Client, handle_file

# Set this to the public URL printed by the Kaggle notebook when you run server.py
KAGGLE_URL = "https://<your-kaggle-gradio-link>.gradio.live"

def execute_payload(payload_json: str):
    import pyautogui
    try:
        data = json.loads(payload_json)
        # ActionSerializer outputs Win32 payloads, but we can extract norm_pos
        metadata = data.get("metadata", {})
        norm_pos = metadata.get("norm_pos", [0.0, 0.0])
        action = metadata.get("action", "")
        
        nx, ny = norm_pos[0], norm_pos[1]
        if nx != 0.0 and ny != 0.0:
            sw, sh = pyautogui.size()
            abs_x = int(nx * sw)
            abs_y = int(ny * sh)
            print(f"[Client Actuator] Moving to ({abs_x}, {abs_y})")
            pyautogui.moveTo(abs_x, abs_y, duration=0.2)
            if action in ["click", "launch_app"]:
                pyautogui.click()
    except Exception as e:
        print(f"Failed to execute payload: {e}")

def main():
    print("=== JEV AI Local Client ===")
    if "your-kaggle-gradio-link" in KAGGLE_URL:
        print("ERROR: Please update KAGGLE_URL in client.py with your active Kaggle server link!")
        return

    client = Client(KAGGLE_URL)
    
    while True:
        input("\nPress ENTER to capture screen, listen to microphone, and send to Kaggle... (or Ctrl+C to exit)")
        
        # 1. Grab Live Screen
        print("Taking screenshot...")
        pil_img = ImageGrab.grab()
        screen_path = "temp_screen.jpg"
        pil_img.save(screen_path)
        
        # 2. Record Audio
        print("Recording 4 seconds of audio...")
        import sounddevice as sd
        import soundfile as sf
        fs = 16000
        duration = 4  # seconds
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        audio_path = "temp_audio.wav"
        sf.write(audio_path, recording, fs)
        
        # 3. Send to Kaggle
        print("Sending to Kaggle GPU Server...")
        t0 = time.time()
        try:
            result = client.predict(
                audio_path=handle_file(audio_path),
                image_path=handle_file(screen_path),
                api_name="/evaluate"
            )
            decision_json, payload_json = result
            print(f"\n[Kaggle Response Time] {time.time() - t0:.2f} seconds")
            
            # Print decision
            dec = json.loads(decision_json)
            print(f"Action: {dec.get('action')} | Target: {dec.get('target')}")
            
            # 4. Execute Locally
            execute_payload(payload_json)
            
        except Exception as e:
            print(f"API Error: {e}")

if __name__ == "__main__":
    main()
