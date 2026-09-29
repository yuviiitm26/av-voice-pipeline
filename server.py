import os
import cv2
import numpy as np
import gradio as gr

from av_pipeline.audio import AVTSEEngine, SileroVADProcessor, StreamingASREngine
from av_pipeline.brain import JevBrainRouter
from av_pipeline.vision import PixelJevGrounder, ActionSerializer

def build_api_server():
    print("Initializing Cloud AI Models...")
    # These models will load onto the Kaggle GPU
    _asr = StreamingASREngine(device="cuda")
    _asr.load_model()
    _jev = JevBrainRouter()
    _grounder = PixelJevGrounder()
    _serializer = ActionSerializer()

    def process_cloud_request(audio_path, image_path):
        if audio_path is None or image_path is None:
            return "{}", "{}"

        # 1. Load the screenshot sent by the client
        screen_img = cv2.imread(image_path)
        if screen_img is None:
            return "{}", "{}"

        # 2. Load and process audio sent by the client
        import soundfile as sf
        import librosa
        audio_f32, sr_in = sf.read(audio_path, dtype="float32")
        if audio_f32.ndim > 1:
            audio_f32 = audio_f32.mean(axis=1)
        if sr_in != 16000:
            audio_f32 = librosa.resample(audio_f32, orig_sr=sr_in, target_sr=16000)

        # 3. Pipeline Execution (Lightning fast on GPU)
        asr_res   = _asr.transcribe(audio_f32)
        decision  = _jev.evaluate(asr_res["text"])
        
        # We pass the raw transcript to the grounder so OCR can find exact text matches
        grounding = _grounder.ground(screen_img, asr_res["text"])
        payload   = _serializer.serialize_decision(decision, grounding)

        return decision.model_dump_json(), payload.model_dump_json()

    # Create a headless Gradio API
    with gr.Blocks() as app:
        gr.Markdown("## JEV AI Cloud Brain API")
        with gr.Row():
            audio_in = gr.Audio(type="filepath")
            image_in = gr.Image(type="filepath")
        with gr.Row():
            decision_out = gr.JSON()
            payload_out = gr.JSON()
            
        btn = gr.Button("Process")
        btn.click(process_cloud_request, inputs=[audio_in, image_in], outputs=[decision_out, payload_out], api_name="evaluate")

    return app

if __name__ == "__main__":
    app = build_api_server()
    app.launch(share=True)  # share=True exposes it to the public internet so the Windows client can connect
