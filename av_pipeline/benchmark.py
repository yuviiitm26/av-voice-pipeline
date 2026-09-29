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

from .synthetic import CocktailPartyMixer
from .audio import AVTSEEngine, SileroVADProcessor, StreamingASREngine
from .brain import JevBrainRouter
from .vision import PixelJevGrounder, ActionSerializer

def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Distortion Ratio (dB)."""
    eps = 1e-8
    ref = reference - np.mean(reference)
    est = estimate - np.mean(estimate)
    ref_energy = np.sum(ref ** 2)
    if ref_energy < eps:
        return 0.0
    alpha = np.sum(ref * est) / (ref_energy + eps)
    s_target = alpha * ref
    e_noise  = est - s_target
    return float(10.0 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)))


def si_sdri(reference: np.ndarray, estimate: np.ndarray, mixture: np.ndarray) -> float:
    """SI-SDR improvement (dB)."""
    return si_sdr(reference, estimate) - si_sdr(reference, mixture)


class PipelineBenchmark:
    """
    Full end-to-end benchmark:
    Raw Mix → AV-TSE → Silero VAD → ASR → Jev Decision → Action Serialization
    """

    def __init__(
        self,
        avtse: AVTSEEngine,
        vad: SileroVADProcessor,
        asr: StreamingASREngine,
        jev: JevBrainRouter,
        grounder: PixelJevGrounder,
        serializer: ActionSerializer,
        num_trials: int = 50,
        sir_db: float = -5.0,
        duration_s: float = 3.0,
    ):
        self.avtse      = avtse
        self.vad        = vad
        self.asr        = asr
        self.jev        = jev
        self.grounder   = grounder
        self.serializer = serializer
        self.num_trials = num_trials
        self.sir_db     = sir_db
        self.duration_s = duration_s
        self._screen    = generate_sample_desktop_screenshot()

    # ──────────────────────────────────────────────────
    def run_single_trial(self, trial_idx: int) -> Dict[str, Any]:
        mixer = CocktailPartyMixer(num_speakers=5, duration_s=self.duration_s)
        mixed, target_clean, lips, _ = mixer.mix(sir_db=self.sir_db)

        timing: Dict[str, Any] = {"trial": trial_idx}

        # 1  AV-TSE
        t0 = time.perf_counter_ns()
        estimated = self.avtse.extract(mixed, lips)
        timing["avtse_ms"] = (time.perf_counter_ns() - t0) / 1e6

        # Audio quality
        min_len = min(len(target_clean), len(estimated), len(mixed))
        timing["si_sdri_db"] = si_sdri(
            target_clean[:min_len], estimated[:min_len], mixed[:min_len],
        )

        # 2  VAD
        t0 = time.perf_counter_ns()
        vad_out = self.vad.process(estimated)
        timing["vad_ms"] = (time.perf_counter_ns() - t0) / 1e6
        self.vad.reset()

        # 3  ASR
        audio_for_asr = vad_out["trimmed_audio"] if len(vad_out["trimmed_audio"]) > 160 else estimated
        t0 = time.perf_counter_ns()
        asr_res = self.asr.transcribe(audio_for_asr)
        timing["asr_ms"] = (time.perf_counter_ns() - t0) / 1e6

        # 4  Jev
        t0 = time.perf_counter_ns()
        decision = self.jev.evaluate(asr_res["text"])
        timing["jev_ms"] = (time.perf_counter_ns() - t0) / 1e6

        # 5  Ground + Serialize
        t0 = time.perf_counter_ns()
        grounding = self.grounder.ground(self._screen, decision.target.value)
        _ = self.serializer.serialize_decision(decision, grounding)
        timing["serialize_ms"] = (time.perf_counter_ns() - t0) / 1e6

        timing["total_ms"] = sum([
            timing["avtse_ms"], timing["vad_ms"], timing["asr_ms"],
            timing["jev_ms"], timing["serialize_ms"],
        ])
        return timing

    # ──────────────────────────────────────────────────
    def run_all(self) -> pd.DataFrame:
        rows = []
        for i in tqdm(range(self.num_trials), desc="Benchmark trials"):
            rows.append(self.run_single_trial(i))
        return pd.DataFrame(rows)

    # ──────────────────────────────────────────────────
    @staticmethod
    def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["avtse_ms", "vad_ms", "asr_ms", "jev_ms", "serialize_ms", "total_ms"]
        stats: Dict[str, Dict[str, float]] = {}
        for c in cols:
            stats[c] = {
                "Mean": df[c].mean(),
                "P50":  df[c].quantile(0.50),
                "P95":  df[c].quantile(0.95),
                "P99":  df[c].quantile(0.99),
            }
        return pd.DataFrame(stats).T

    # ──────────────────────────────────────────────────
    @staticmethod
    def display_results(df: pd.DataFrame, stats: pd.DataFrame) -> Tuple[plt.Figure, plt.Figure]:
        print("\n" + "=" * 60)
        print("END-TO-END LATENCY STATISTICS (ms)".center(60))
        print("=" * 60)
        print(stats.round(2).to_string())
        print("-" * 60)
        print(f"Mean SI-SDRi : {df['si_sdri_db'].mean():.2f} dB")
        print("=" * 60)

        # Fig 1 – Latency histogram
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.hist(df["total_ms"], bins=20, color="skyblue", edgecolor="black")
        ax1.axvline(df["total_ms"].median(), color="red", linestyle="--", label="Median")
        ax1.set_title("Total Pipeline Latency Distribution")
        ax1.set_xlabel("Latency (ms)"); ax1.set_ylabel("Count")
        ax1.legend(); fig1.tight_layout()

        # Fig 2 – Stacked breakdown
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        stages  = ["avtse_ms", "vad_ms", "asr_ms", "jev_ms", "serialize_ms"]
        labels  = ["AV-TSE", "VAD", "ASR", "Jev", "Ground+Serialize"]
        colours = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]
        bottom = 0.0
        for stage, label, col in zip(stages, labels, colours):
            val = df[stage].mean()
            ax2.bar(["Pipeline"], [val], bottom=bottom, label=label, color=col)
            bottom += val
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Mean Latency Breakdown")
        ax2.legend(loc="upper left"); fig2.tight_layout()

        return fig1, fig2


