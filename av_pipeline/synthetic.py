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

@dataclass
class SpeakerProfile:
    """Configuration profile for one synthesized speaker."""
    speaker_id: str
    frequency_base: float
    formants: List[float]
    amplitude: float = 1.0


def generate_synthetic_speech(
    profile: SpeakerProfile,
    duration_s: float,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """
    Produce a synthetic speech-like waveform by summing formant
    sine waves with amplitude modulation, frequency jitter, and
    random pauses.

    Returns:
        1-D float32 ndarray normalised to [-1, 1].
    """
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    out = np.zeros(n_samples, dtype=np.float64)

    # Multi-formant synthesis with per-formant tremolo jitter
    for f in profile.formants:
        jitter_hz = 4.0 + np.random.rand() * 2.0
        jitter = np.sin(2 * np.pi * jitter_hz * t) * 0.01 * f
        out += np.sin(2 * np.pi * (f + jitter) * t)

    # Syllabic envelope (3–5 Hz AM)
    syl_rate = 3.0 + np.random.rand() * 2.0
    base_env = np.abs(np.sin(2 * np.pi * syl_rate * t))

    # Smooth noise envelope for naturalness
    env_noise = np.random.randn(n_samples)
    b, a = sig.butter(3, 10.0 / (sample_rate / 2.0), btype="low")
    smooth_env = sig.filtfilt(b, a, np.abs(env_noise))
    smooth_env -= smooth_env.min()
    smooth_env /= (smooth_env.max() + 1e-8)

    env = base_env * 0.6 + smooth_env * 0.4
    out *= env * profile.amplitude

    # Random pauses
    for _ in range(max(1, int(duration_s))):
        ps = int(np.random.rand() * n_samples)
        pl = int((0.10 + np.random.rand() * 0.30) * sample_rate)
        pe = min(ps + pl, n_samples)
        fade = min(int(0.01 * sample_rate), (pe - ps) // 2)
        if fade > 0 and pe - ps > 2 * fade:
            win = np.ones(pe - ps)
            win[:fade] = np.linspace(1, 0, fade)
            win[-fade:] = np.linspace(0, 1, fade)
            win[fade:-fade] = 0.0
            out[ps:pe] *= win

    mx = np.max(np.abs(out))
    if mx > 0:
        out /= mx
    return out.astype(np.float32)


def generate_lip_video_crops(
    num_frames: int,
    fps: int = 25,
    size: int = 96,
) -> np.ndarray:
    """
    Synthesise grayscale 96×96 lip-region crops with a rhythmically
    opening/closing mouth (ellipse).

    Returns:
        (T, 1, size, size) float32 tensor in [0, 1].
    """
    syl_rate = 3.0  # Hz
    frames = np.zeros((num_frames, 1, size, size), dtype=np.float32)
    center = (size // 2, size // 2)

    for i in range(num_frames):
        canvas = np.zeros((size, size), dtype=np.uint8)
        t = i / fps
        open_ratio = (np.sin(2 * np.pi * syl_rate * t) + 1.0) / 2.0
        axes = (int(size * 0.30), int(size * 0.05 + size * 0.20 * open_ratio))
        cv2.ellipse(canvas, center, axes, 0, 0, 360, 255, -1)
        frames[i, 0] = canvas.astype(np.float32) / 255.0

    return frames


class CocktailPartyMixer:
    """Generates a configurable multi-speaker cocktail-party scenario."""

    def __init__(
        self,
        num_speakers: int = 5,
        duration_s: float = 5.0,
        sample_rate: int = 16_000,
    ):
        self.num_speakers = max(2, num_speakers)
        self.duration_s = duration_s
        self.sample_rate = sample_rate

    # ──────────────────────────────────────────────────
    def generate_profiles(self) -> List[SpeakerProfile]:
        profiles: List[SpeakerProfile] = []
        for i in range(self.num_speakers):
            base = 120.0 + i * 45.0
            profiles.append(SpeakerProfile(
                speaker_id=f"Speaker_{i}",
                frequency_base=base,
                formants=[base, base * 2.2, base * 3.4, base * 4.5],
                amplitude=1.0 - i * 0.05,
            ))
        return profiles

    # ──────────────────────────────────────────────────
    def mix(
        self,
        sir_db: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Returns
        -------
        mixed     – (N,) float32
        target    – (N,) float32
        lip_crops – (T, 1, 96, 96) float32
        sources   – list of individual speaker waveforms
        """
        profiles = self.generate_profiles()
        sources = [
            generate_synthetic_speech(p, self.duration_s, self.sample_rate)
            for p in profiles
        ]
        target = sources[0].copy()

        interferers = np.sum(sources[1:], axis=0)
        rms_t = np.sqrt(np.mean(target ** 2) + 1e-10)
        rms_i = np.sqrt(np.mean(interferers ** 2) + 1e-10)

        desired_rms_i = rms_t / (10 ** (sir_db / 20.0))
        interferers *= desired_rms_i / rms_i

        mixed = target + interferers
        mx = np.max(np.abs(mixed))
        if mx > 0:
            scale = 1.0 / mx
            mixed *= scale
            target *= scale
            sources = [s * scale for s in sources]

        fps = 25
        num_frames = int(self.duration_s * fps)
        lip_crops = generate_lip_video_crops(num_frames, fps, size=96)

        return mixed, target, lip_crops, sources

    # ──────────────────────────────────────────────────
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "num_speakers": self.num_speakers,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
        }


def visualize_cocktail_party(
    mixed: np.ndarray,
    clean: np.ndarray,
    lip_crops: np.ndarray,
    sr: int = 16_000,
) -> plt.Figure:
    """
    Side-by-side waveforms, spectrogram, and lip-crop strip.
    Returns the matplotlib Figure for Gradio compatibility.
    """
    fig = plt.figure(figsize=(16, 12))

    # 1 ── Waveforms
    ax1 = fig.add_subplot(3, 1, 1)
    t = np.linspace(0, len(mixed) / sr, len(mixed))
    ax1.plot(t, mixed, alpha=0.55, label="Mixed (5-speaker)", color="orange")
    ax1.plot(t, clean, alpha=0.80, label="Clean Target", color="steelblue")
    ax1.set_title("Waveforms"); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right"); ax1.grid(True, alpha=0.3)

    # 2 ── Spectrogram of mix
    ax2 = fig.add_subplot(3, 1, 2)
    f, ts, Sxx = sig.spectrogram(mixed, fs=sr, nperseg=256, noverlap=128)
    im = ax2.pcolormesh(ts, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis")
    ax2.set_title("Spectrogram (Mixed)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax2, label="dB")

    # 3 ── Lip crop strip
    ax3 = fig.add_subplot(3, 1, 3)
    strip = lip_crops[::5][:12]  # every 5th frame, up to 12
    concat = np.concatenate([fr[0] for fr in strip], axis=1)
    ax3.imshow(concat, cmap="gray"); ax3.set_title("Lip Crops (sampled)"); ax3.axis("off")

    fig.tight_layout()
    return fig


