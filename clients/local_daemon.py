import sys
import os
# Ensure the root directory is in the path so av_pipeline imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Import our local pipeline components
from av_pipeline.audio import StreamingASREngine
from av_pipeline.brain import JevBrainRouter
from av_pipeline.vision import PixelJevGrounder, ActionSerializer

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
    print("=== JEV AI 100% Offline Continuous Daemon ===")
    print("Loading heavy AI models into your local CPU (this takes a moment)...")
    
    _asr = StreamingASREngine(device="cpu")
    _asr.load_model()
    _jev = JevBrainRouter()
    _grounder = PixelJevGrounder()
    _serializer = ActionSerializer()
    
    print("Models loaded successfully!")
    
    print(f"Listening automatically in the background... (Threshold: {SILENCE_THRESHOLD})")
    
    # Let sounddevice automatically negotiate the best hardware format (avoids MME crash)
    stream = sd.InputStream(callback=audio_callback)
    with stream:
        fs = int(stream.samplerate)
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
                            print("[VAD] Silence detected. Processing command locally...")
                            break
                    else:
                        silence_frames = 0
            
            t0 = time.time()
            
            # 1. Grab Screen
            pil_img = ImageGrab.grab()
            screen_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            # 2. Process Audio
            audio_data = np.concatenate(recording, axis=0)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1) # mix to mono
            
            # Resample to 16k for Whisper
            if fs != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=16000)
                
            # Normalize
            mx = np.max(np.abs(audio_data))
            if mx > 0:
                audio_data /= mx
                
            # 3. Local Execution!
            print(">> [1/3] Transcribing audio with Whisper...")
            asr_res   = _asr.transcribe(audio_data)
            print(f"   Transcript: '{asr_res['text']}'")
            
            print(">> [2/3] Parsing intent...")
            decision  = _jev.evaluate(asr_res["text"])
            print(f"   Action: {decision.action.value} on {decision.target.value}")
            
            print(">> [3/3] Running YOLO/OCR Vision Grounder on your screen...")
            grounding = _grounder.ground(screen_img, asr_res["text"])
            payload   = _serializer.serialize_decision(decision, grounding)
            
            print(f"[Local CPU Latency] {time.time() - t0:.2f}s")
            
            # 4. Actuate
            execute_payload(payload.model_dump_json())
            print("\nListening for next command...")

if __name__ == "__main__":
    main()
