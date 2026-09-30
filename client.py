import os
import cv2
import json
import time
import queue
import numpy as np
import threading
import sounddevice as sd
import soundfile as sf
from PIL import ImageGrab
from gradio_client import Client, handle_file

KAGGLE_URL = "https://14549f5d37961c9352.gradio.live"
SILENCE_THRESHOLD = 0.015  # Adjust this if it's too sensitive or not sensitive enough
SILENCE_DURATION = 1.5     # Seconds of silence before stopping

audio_queue = queue.Queue()

def execute_payload(payload_json: str):
    import pyautogui
    try:
        data = json.loads(payload_json)
        metadata = data.get("metadata", {})
        norm_pos = metadata.get("norm_pos", [0.0, 0.0])
        action = metadata.get("action", "")
        
        nx, ny = norm_pos[0], norm_pos[1]
        if nx != 0.0 and ny != 0.0:
            sw, sh = pyautogui.size()
            abs_x = int(nx * sw)
            abs_y = int(ny * sh)
            print(f"[Actuator] Executing click at ({abs_x}, {abs_y})")
            pyautogui.moveTo(abs_x, abs_y, duration=0.2)
            if action in ["click", "launch_app"]:
                pyautogui.click()
    except Exception as e:
        pass

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def main():
    print("=== JEV AI Continuous Client ===")
    print(f"Connecting to Kaggle AI Brain at: {KAGGLE_URL}")
    client = Client(KAGGLE_URL)
    
    device_info = sd.query_devices(sd.default.device[0], 'input')
    fs = int(device_info.get('default_samplerate', 44100))
    channels = min(2, device_info.get('max_input_channels', 1))

    print(f"Listening automatically... (Threshold: {SILENCE_THRESHOLD})")
    
    with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
        while True:
            # Wait for someone to start speaking
            recording = []
            is_speaking = False
            silence_frames = 0
            
            while True:
                chunk = audio_queue.get()
                rms = np.sqrt(np.mean(chunk**2))
                
                if not is_speaking:
                    if rms > SILENCE_THRESHOLD:
                        print("\n[VAD] Speech detected! Recording...")
                        is_speaking = True
                        recording.append(chunk)
                else:
                    recording.append(chunk)
                    if rms < SILENCE_THRESHOLD:
                        silence_frames += len(chunk)
                        if (silence_frames / fs) > SILENCE_DURATION:
                            print("[VAD] Silence detected. Processing command...")
                            break
                    else:
                        silence_frames = 0
            
            # 1. Grab Screen
            pil_img = ImageGrab.grab()
            screen_path = "temp_screen.jpg"
            pil_img.save(screen_path)
            
            # 2. Save Audio
            audio_data = np.concatenate(recording, axis=0)
            audio_path = "temp_audio.wav"
            sf.write(audio_path, audio_data, fs)
            
            # 3. Send to Kaggle
            print(">> Sending to Cloud Brain...")
            t0 = time.time()
            try:
                result = client.predict(
                    audio_path=handle_file(audio_path),
                    image_path=handle_file(screen_path),
                    api_name="/evaluate"
                )
                decision_json, payload_json = result
                print(f"[Brain] Latency: {time.time() - t0:.2f}s")
                dec = json.loads(decision_json)
                print(f"   -> {dec.get('action')} on {dec.get('target')}")
                
                execute_payload(payload_json)
            except Exception as e:
                print(f"[Error] Failed to connect: {e}")
                
            print("\nListening for next command...")

if __name__ == "__main__":
    main()
