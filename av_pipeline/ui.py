from __future__ import annotations
import os
import sys
import math
import json
import time
import uuid
import platform
import warnings
from enum import Enum, IntEnum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import cv2
import scipy.signal as sig
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from pydantic import BaseModel, Field

from .core import ActionType
from .diagnostics import run_diagnostics
from .synthetic import generate_synthetic_speech, generate_lip_video_crops, CocktailPartyMixer, visualize_cocktail_party
from .audio import AVTSEEngine, SileroVADProcessor, StreamingASREngine
from .brain import JevBrainRouter
from .vision import PixelJevGrounder, ActionSerializer, generate_sample_desktop_screenshot
from .benchmark import PipelineBenchmark

def build_gradio_app(
    avtse: AVTSEEngine,
    vad: SileroVADProcessor,
    asr: StreamingASREngine,
    jev: JevBrainRouter,
    grounder: PixelJevGrounder,
    serializer: ActionSerializer,
) -> gr.Blocks:
    """Constructs a 3-tab Gradio Blocks application."""

    # No dummy screen here; we grab the live screen inside the functions when needed.
    from PIL import ImageGrab

    # ────────────────── Tab 1: Synthetic ──────────────────
    def process_synthetic(
        num_speakers: int,
        sir_db: float,
        duration_s: float,
    ):
        mixer = CocktailPartyMixer(
            num_speakers=int(num_speakers),
            duration_s=float(duration_s),
        )
        mixed, target, lips, _ = mixer.mix(sir_db=float(sir_db))

        # AV-TSE
        separated = avtse.extract(mixed, lips)

        # VAD
        vad_out = vad.process(separated)
        vad.reset()
        audio_for_asr = vad_out["trimmed_audio"] if len(vad_out["trimmed_audio"]) > 160 else separated

        # ASR
        asr_res = asr.transcribe(audio_for_asr)
        transcript = asr_res["text"]

        # Decision
        decision = jev.evaluate(transcript)

        # Grounding & serialization
        grounding = grounder.ground(screen_img, decision.target.value)
        payload = serializer.serialize_decision(decision, grounding)

        # Waveform figure
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        t = np.linspace(0, len(mixed) / 16000, len(mixed))
        axes[0].plot(t, mixed, alpha=0.7); axes[0].set_title("Mixed"); axes[0].set_ylabel("Amp")
        t2 = np.linspace(0, len(separated) / 16000, len(separated))
        axes[1].plot(t2, separated, alpha=0.7, color="green"); axes[1].set_title("Separated")
        axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Amp")
        fig.tight_layout()

        # Return audio as (sr, ndarray) tuples for gr.Audio
        return (
            (16000, mixed),            # mixed audio
            (16000, separated),        # separated audio
            fig,                       # waveform plot
            transcript,                # transcript text
            decision.model_dump_json(indent=2),  # Jev JSON
            payload.model_dump_json(indent=2),   # payload JSON
        )

    # ────────────────── Tab 2: Live Microphone ──────────────────
    def process_live(audio_input):
        if audio_input is None:
            return "No audio recorded.", "{}", "{}"

        # audio_input is (sample_rate, ndarray) from Gradio
        if isinstance(audio_input, tuple):
            sr_in, audio_data = audio_input
            audio_f32 = audio_data.astype(np.float32)
            if audio_f32.ndim > 1:
                audio_f32 = audio_f32.mean(axis=1)
            # Normalise
            mx = np.max(np.abs(audio_f32))
            if mx > 0:
                audio_f32 /= mx
        else:
            # filepath fallback
            try:
                import soundfile as sf
                audio_f32, sr_in = sf.read(audio_input, dtype="float32")
                if audio_f32.ndim > 1:
                    audio_f32 = audio_f32.mean(axis=1)
            except Exception:
                return "Failed to load audio.", "{}", "{}"

        # Resample to 16k if needed
        if sr_in != 16000:
            import librosa
            audio_f32 = librosa.resample(audio_f32, orig_sr=sr_in, target_sr=16000)

        asr_res   = asr.transcribe(audio_f32)
        decision  = jev.evaluate(asr_res["text"])

        # Capture live screen for OCR!
        pil_img = ImageGrab.grab()
        screen_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Ground against the raw transcript so OCR can find the exact words!
        grounding = grounder.ground(screen_img, asr_res["text"])
        payload   = serializer.serialize_decision(decision, grounding)

        # ====================================================
        # GENERIC LOCAL EXECUTION HOOK (Visible Demo for User)
        # ====================================================
        transcript_lower = asr_res["text"].lower()
        print(f"\n[ASR Transcript] {transcript_lower}")
        print(f"[Decision] {decision.action.value} on {decision.target.value}")
        
        # We execute generic clicks if YOLO/OCR found coordinates
        if decision.action != ActionType.NO_ACTION:
            try:
                import pyautogui
                nx = grounding.get("norm_x", 0.0)
                ny = grounding.get("norm_y", 0.0)
                
                # if we have a valid coordinate (not 0,0 default fallback)
                if nx != 0.0 and ny != 0.0:
                    sw, sh = pyautogui.size()
                    abs_x = int(nx * sw)
                    abs_y = int(ny * sh)
                    print(f"[Actuator] Moving to ({abs_x}, {abs_y}) and clicking...")
                    pyautogui.moveTo(abs_x, abs_y, duration=0.5)
                    pyautogui.click()
                
                # If they want to type
                if decision.action == ActionType.TYPE_TEXT:
                    # simplistic extraction: type whatever comes after "type" or "write"
                    words = transcript_lower.split()
                    for kw in ["type", "write", "enter"]:
                        if kw in words:
                            idx = words.index(kw)
                            text_to_type = " ".join(words[idx+1:])
                            if text_to_type:
                                print(f"[Actuator] Typing: {text_to_type}")
                                pyautogui.write(text_to_type, interval=0.05)
                            break
            except ImportError:
                print("[Actuator] pyautogui not installed, skipping physical execution.")

        return (
            asr_res["text"],
            decision.model_dump_json(indent=2),
            payload.model_dump_json(indent=2),
        )

    # ────────────────── Tab 3: Benchmark ──────────────────
    def run_benchmark(num_trials: int):
        bench = PipelineBenchmark(
            avtse=avtse, vad=vad, asr=asr, jev=jev,
            grounder=grounder, serializer=serializer,
            num_trials=int(num_trials), sir_db=-5.0, duration_s=3.0,
        )
        df = bench.run_all()
        stats = PipelineBenchmark.compute_statistics(df)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["total_ms"], bins=20, color="coral", edgecolor="black")
        ax.axvline(df["total_ms"].median(), color="navy", linestyle="--", label="Median")
        ax.set_title("Total Pipeline Latency Distribution")
        ax.set_xlabel("ms"); ax.set_ylabel("Count"); ax.legend()
        fig.tight_layout()

        stats_display = stats.reset_index().rename(columns={"index": "Stage"}).round(2)
        return stats_display, fig

    # ────────────────── Build UI ──────────────────
    with gr.Blocks(
        title="AV Voice Automation Pipeline",
    ) as app:
        gr.Markdown(
            "# Ultra-Low-Latency Audio-Visual Voice Automation Pipeline\n"
            "End-to-end demo: **Mix -> AV-TSE -> VAD -> ASR -> Jev -> Win32 Payload**"
        )

        with gr.Tab("Synthetic Pipeline Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_speakers = gr.Slider(2, 8, value=5, step=1, label="Number of Speakers")
                    sl_sir      = gr.Slider(-15, 10, value=-5, step=1, label="SIR (dB)")
                    sl_dur      = gr.Slider(1, 10, value=3, step=0.5, label="Duration (s)")
                    btn_gen     = gr.Button("Generate & Process", variant="primary")
                with gr.Column(scale=2):
                    out_mix = gr.Audio(label="Mixed Audio (5-speaker)")
                    out_sep = gr.Audio(label="Separated (target speaker)")
                    out_plt = gr.Plot(label="Waveforms")
            out_txt = gr.Textbox(label="Transcript", lines=2)
            with gr.Row():
                out_jev = gr.Code(label="Jev Decision (JSON)", language="json")
                out_pay = gr.Code(label="Action Payload (JSON)", language="json")
            btn_gen.click(
                process_synthetic,
                inputs=[sl_speakers, sl_sir, sl_dur],
                outputs=[out_mix, out_sep, out_plt, out_txt, out_jev, out_pay],
            )

        with gr.Tab("Live Microphone"):
            gr.Markdown("Record a voice command via the browser microphone.")
            with gr.Row():
                mic_in   = gr.Audio(sources=["microphone"], label="Record Voice Command")
                btn_live = gr.Button("Process Command", variant="primary")
            live_txt = gr.Textbox(label="Transcript", lines=2)
            with gr.Row():
                live_jev = gr.Code(label="Jev Decision (JSON)", language="json")
                live_pay = gr.Code(label="Action Payload (JSON)", language="json")
            btn_live.click(
                process_live,
                inputs=[mic_in],
                outputs=[live_txt, live_jev, live_pay],
            )

        with gr.Tab("Benchmark"):
            gr.Markdown("Run the full pipeline benchmark across N trials.")
            sl_trials = gr.Slider(10, 100, value=50, step=10, label="Number of Trials")
            btn_bench = gr.Button("Run Benchmark", variant="primary")
            bench_tbl = gr.Dataframe(label="Latency Statistics (ms)")
            bench_plt = gr.Plot(label="Latency Distribution")
            btn_bench.click(
                run_benchmark,
                inputs=[sl_trials],
                outputs=[bench_tbl, bench_plt],
            )

    return app


