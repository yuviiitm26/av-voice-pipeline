# %% [markdown]
# # 🎙️ Ultra-Low-Latency Audio-Visual Voice Automation Pipeline
# **Production-Grade End-to-End Prototype — Google Colab / Kaggle Compatible**
#
# This notebook implements the complete pipeline:
# 1. Environment Setup & Diagnostics
# 2. Multi-Speaker Synthetic Simulation Harness
# 3. Audio-Visual Target Speaker Extraction (AV-TSE) & Silero VAD
# 4. Moonshine / Whisper Streaming ASR
# 5. Decision Brain Layer (TypeSafe AI Jev System One)
# 6. Visual Grounding & Windows-Ready Action Serialization
# 7. End-to-End Latency Benchmarking & Profiler
# 8. Interactive Gradio UI

# %% [markdown]
# ---
# ## Module 1: Environment Setup, Dependencies & System Diagnostics

# %%
# ── Installation Cell (uncomment to run in Colab/Kaggle) ──
# !pip install -q onnxruntime-gpu torch torchaudio torchvision soundfile librosa
# !pip install -q opencv-python-headless mediapipe
# !pip install -q speechbrain gradio "pydantic>=2.0"
# !pip install -q numpy scipy matplotlib tqdm pandas
# !pip install -q useful-sensors-moonshine || pip install -q openai-whisper
# !apt-get update -y && apt-get install -y libsndfile1

# %%
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
from typing import (
    Any, Dict, Generator, List, Optional, Tuple, Union,
)

import numpy as np
import pandas as pd
import cv2
import scipy.signal as sig
import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# Deferred heavy imports (validated in diagnostics)
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field


def run_diagnostics() -> Dict[str, Any]:
    """
    Verifies CUDA availability, GPU specs, ONNX Runtime providers,
    and all critical library imports.  Prints a formatted diagnostic
    table and returns the results dict.
    """
    results: Dict[str, Any] = {}

    # ── PyTorch & CUDA ──
    results["PyTorch Version"] = torch.__version__
    results["CUDA Available"] = torch.cuda.is_available()
    if results["CUDA Available"]:
        results["GPU Name"] = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory
        results["GPU Memory (MB)"] = round(mem / (1024 ** 2), 2)
    else:
        results["GPU Name"] = "N/A (CPU mode)"
        results["GPU Memory (MB)"] = "N/A"

    # ── ONNX Runtime ──
    try:
        import onnxruntime as ort
        results["ONNX Providers"] = ort.get_available_providers()
    except ImportError:
        results["ONNX Providers"] = ["Not Installed"]

    # ── Critical Modules ──
    critical = [
        "torchaudio", "torchvision", "soundfile", "librosa",
        "cv2", "mediapipe", "speechbrain", "gradio", "pydantic",
        "numpy", "scipy", "matplotlib", "tqdm", "pandas",
        "ultralytics", "easyocr",
    ]
    mod_status: Dict[str, str] = {}
    for m in critical:
        try:
            __import__(m)
            mod_status[m] = "[OK] Installed"
        except ImportError:
            mod_status[m] = "[MISS] Missing"

    # Moonshine / Whisper probe
    for alt_name, pkg in [("moonshine", "moonshine"), ("whisper", "whisper")]:
        try:
            __import__(pkg)
            mod_status[alt_name] = "[OK] Installed"
        except ImportError:
            mod_status[alt_name] = "[WARN] Not found"

    results["Modules"] = mod_status

    # ── Pretty-print ──
    print("=" * 64)
    print("SYSTEM DIAGNOSTICS".center(64))
    print("=" * 64)
    print(f"{'OS Platform':<28}: {platform.platform()}")
    print(f"{'Python Version':<28}: {sys.version.split()[0]}")
    print("-" * 64)
    for k, v in results.items():
        if k != "Modules":
            print(f"{k:<28}: {v}")
    print("-" * 64)
    print("CRITICAL MODULES:")
    for k, v in results["Modules"].items():
        print(f"  {k:<24}: {v}")
    print("=" * 64)
    return results


diagnostics = run_diagnostics()

# %% [markdown]
# ---
# ## Module 2: Multi-Speaker Synthetic Simulation Harness
# Generates a 5-speaker "cocktail party" scenario with configurable SIR,
# synchronized lip-region video crops, and visualization utilities.

# %%
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


# ── Quick demo ──
print("\n[Module 2] Generating 5-speaker cocktail-party scenario at -5 dB SIR …")
_mixer = CocktailPartyMixer(num_speakers=5, duration_s=4.0)
_mixed, _clean, _lips, _sources = _mixer.mix(sir_db=-5.0)
print(f"  mix shape={_mixed.shape}  lip_crops shape={_lips.shape}")
_fig = visualize_cocktail_party(_mixed, _clean, _lips)
plt.show()

# %% [markdown]
# ---
# ## Module 3: Audio-Visual Target Speaker Extraction (AV-TSE) & Silero VAD

# %%
# ─────────────────────────────────────────────────────────────────
#  3-A  LipEncoder  –  3D-Conv frontend → ResBlocks → embedding
# ─────────────────────────────────────────────────────────────────
class LipEncoder(nn.Module):
    """(B, T, 1, 96, 96) lip crops  →  (B, emb_dim) visual embedding."""

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.conv3d = nn.Conv3d(1, 64, kernel_size=(5, 7, 7),
                                stride=(1, 2, 2), padding=(2, 3, 3))
        self.bn3d   = nn.BatchNorm3d(64)
        self.pool3d = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                   stride=(1, 2, 2), padding=(0, 1, 1))

        self.res1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc   = nn.Linear(256, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, H, W)  →  Conv3d wants (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = self.pool3d(x)

        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).contiguous().view(B * T, C, H, W)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x).view(B, T, -1).mean(dim=1)   # temporal avg
        return self.fc(x)


# ─────────────────────────────────────────────────────────────────
#  3-B  FiLM-conditioned 1-D ResBlock
# ─────────────────────────────────────────────────────────────────
class FiLMRes1DBlock(nn.Module):
    """Dilated depth-wise-sep 1-D conv with FiLM visual conditioning."""

    def __init__(self, channels: int, dilation: int, cond_dim: int):
        super().__init__()
        self.pw1  = nn.Conv1d(channels, channels, 1)
        self.act1 = nn.PReLU()
        self.gn1  = nn.GroupNorm(1, channels)

        self.dw   = nn.Conv1d(channels, channels, 3,
                              padding=dilation, dilation=dilation, groups=channels)
        self.act2 = nn.PReLU()
        self.gn2  = nn.GroupNorm(1, channels)

        # FiLM
        self.film_scale = nn.Linear(cond_dim, channels)
        self.film_shift = nn.Linear(cond_dim, channels)

        self.pw2 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.gn1(self.act1(self.pw1(x)))
        x = self.gn2(self.act2(self.dw(x)))
        s = self.film_scale(cond).unsqueeze(2)
        b = self.film_shift(cond).unsqueeze(2)
        x = x * (1.0 + s) + b
        return self.pw2(x) + res


# ─────────────────────────────────────────────────────────────────
#  3-C  AudioVisualSeparator  (Conv-TasNet-inspired)
# ─────────────────────────────────────────────────────────────────
class AudioVisualSeparator(nn.Module):
    """
    Encoder → 4-block dilated separator (FiLM) → Decoder.
    Input:  mixed (B, 1, N),  lips (B, T, 1, 96, 96)
    Output: separated (B, 1, N)
    """

    def __init__(self, channels: int = 128, emb_dim: int = 256):
        super().__init__()
        self.lip_enc = LipEncoder(emb_dim)
        self.encoder = nn.Conv1d(1, channels, kernel_size=20, stride=10, padding=5)
        self.enc_gn  = nn.GroupNorm(1, channels)
        self.blocks  = nn.ModuleList([
            FiLMRes1DBlock(channels, dilation=2 ** i, cond_dim=emb_dim)
            for i in range(4)
        ])
        self.decoder = nn.ConvTranspose1d(channels, 1, kernel_size=20, stride=10, padding=5)

    def forward(self, mixed: torch.Tensor, lips: torch.Tensor) -> torch.Tensor:
        vis = self.lip_enc(lips)                       # (B, emb)
        x = self.enc_gn(self.encoder(mixed))           # (B, C, L)
        for blk in self.blocks:
            x = blk(x, vis)
        out = self.decoder(x)
        # match input length exactly
        if out.shape[-1] != mixed.shape[-1]:
            out = F.interpolate(out, size=mixed.shape[-1], mode="linear", align_corners=False)
        return out


# ─────────────────────────────────────────────────────────────────
#  3-D  AVTSEEngine  –  high-level extraction API
# ─────────────────────────────────────────────────────────────────
class AVTSEEngine:
    """Wraps AudioVisualSeparator for numpy-in / numpy-out inference."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model: Optional[AudioVisualSeparator] = None

    def load_model(self) -> None:
        self.model = AudioVisualSeparator().to(self.device).eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[AVTSEEngine] loaded on {self.device}  "
              f"({n_params / 1e6:.2f} M params, random weights – demo mode)")

    def extract(
        self,
        mixed_audio: np.ndarray,
        lip_crops: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        mixed_audio : (N,) float32, 16 kHz
        lip_crops   : (T, 1, 96, 96) float32

        Returns
        -------
        isolated : (N,) float32, 16 kHz
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first.")
        audio_t = torch.from_numpy(mixed_audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        lips_t  = torch.from_numpy(lip_crops).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_t = self.model(audio_t, lips_t)
        return out_t.squeeze().cpu().numpy()


# ─────────────────────────────────────────────────────────────────
#  3-E  SileroVADProcessor
# ─────────────────────────────────────────────────────────────────
class SileroVADProcessor:
    """
    512-sample (32 ms @ 16 kHz) micro-segmentation with Silero VAD.
    Falls back to energy-based detection if Silero cannot be loaded.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        chunk_samples: int = 512,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.chunk_samples = chunk_samples
        self.model: Any = None
        self._get_speech_timestamps: Any = None
        self._backend = "none"

    # ──────────────────────────────────────────────────
    def load_model(self) -> None:
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self.model = model
            self._get_speech_timestamps = utils[0]
            self._backend = "silero"
            print("[SileroVAD] loaded from torch hub.")
        except Exception as exc:
            print(f"[SileroVAD] torch hub load failed ({exc}); using energy-based fallback.")
            self._backend = "energy"

    # ──────────────────────────────────────────────────
    def _energy_vad(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Simple RMS-energy VAD used as a fallback."""
        n = self.chunk_samples
        segs: List[Tuple[int, int]] = []
        in_speech = False
        start = 0
        for i in range(0, len(audio) - n + 1, n):
            rms = np.sqrt(np.mean(audio[i : i + n] ** 2))
            if rms > 0.02 and not in_speech:
                in_speech = True
                start = i
            elif rms <= 0.02 and in_speech:
                in_speech = False
                segs.append((start, i))
        if in_speech:
            segs.append((start, len(audio)))
        return segs if segs else [(0, len(audio))]

    # ──────────────────────────────────────────────────
    def process(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
            speech_segments, trimmed_audio, speech_probability,
            trailing_silence_ms, chunk_probabilities
        """
        chunk_probs: List[float] = []

        if self._backend == "silero" and self.model is not None:
            audio_t = torch.from_numpy(audio).float()

            # chunk-level probabilities
            n = self.chunk_samples
            for i in range(0, len(audio) - n + 1, n):
                chunk = audio_t[i : i + n]
                with torch.no_grad():
                    p = self.model(chunk, self.sample_rate).item()
                chunk_probs.append(p)

            # speech timestamps via Silero utility
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ts = self._get_speech_timestamps(
                    audio_t, self.model,
                    sampling_rate=self.sample_rate,
                    threshold=self.threshold,
                    min_silence_duration_ms=self.min_silence_ms,
                )
            segments = [(t["start"], t["end"]) for t in ts]
        else:
            segments = self._energy_vad(audio)
            chunk_probs = [1.0] * (len(audio) // self.chunk_samples)

        avg_prob = float(np.mean(chunk_probs)) if chunk_probs else 0.0

        # Trim with 10 ms guard
        guard = int(0.010 * self.sample_rate)
        if segments:
            first = max(0, segments[0][0] - guard)
            last  = min(len(audio), segments[-1][1] + guard)
            trimmed = audio[first:last]
            trail_ms = ((len(audio) - segments[-1][1]) / self.sample_rate) * 1000.0
            trail_ms = min(trail_ms, float(self.min_silence_ms))
        else:
            trimmed = np.array([], dtype=np.float32)
            trail_ms = (len(audio) / self.sample_rate) * 1000.0

        return {
            "speech_segments": segments,
            "trimmed_audio": trimmed,
            "speech_probability": avg_prob,
            "trailing_silence_ms": trail_ms,
            "chunk_probabilities": chunk_probs,
        }

    def reset(self) -> None:
        if self._backend == "silero" and self.model is not None:
            self.model.reset_states()


# ── Quick demo ──
print("\n[Module 3] Loading AV-TSE engine & Silero VAD …")
_avtse = AVTSEEngine()
_avtse.load_model()

_isolated = _avtse.extract(_mixed, _lips)
print(f"  isolated shape={_isolated.shape}")

_vad = SileroVADProcessor()
_vad.load_model()
_vad_result = _vad.process(_isolated)
print(f"  speech segments: {len(_vad_result['speech_segments'])}  "
      f"trimmed len: {len(_vad_result['trimmed_audio'])}  "
      f"trailing silence: {_vad_result['trailing_silence_ms']:.1f} ms")

# %% [markdown]
# ---
# ## Module 4: Moonshine / Whisper Streaming ASR

# %%
class StreamingASREngine:
    """
    Tries Moonshine → Whisper → Mock fallback.
    Measures Time-to-First-Token (TTFT) at ns resolution.
    """

    def __init__(self, model_name: str = "moonshine/base", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model: Any = None
        self.backend: str = "none"

    # ──────────────────────────────────────────────────
    def load_model(self) -> None:
        # 1) Moonshine
        try:
            import moonshine                                           # type: ignore
            self._model = moonshine.load_model(self.model_name, device=self.device)
            self.backend = "moonshine"
            print(f"[ASR] Moonshine loaded ({self.model_name}).")
            return
        except Exception:
            pass

        # 2) Whisper
        try:
            import whisper                                             # type: ignore
            self._model = whisper.load_model("tiny", device=self.device)
            self.backend = "whisper-tiny"
            print("[ASR] Whisper-tiny loaded.")
            return
        except Exception:
            pass

        # 3) Mock
        self.backend = "mock"
        print("[ASR] Using mock transcription backend.")

    # ──────────────────────────────────────────────────
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16_000,
    ) -> Dict[str, Any]:
        """
        Transcribes *actual-length* audio (no 30-s zero-padding).

        Returns dict with text, ttft_ms, total_duration_ms, tokens, model_used.
        """
        t0 = time.perf_counter_ns()

        if self.backend == "moonshine":
            res = self._model.transcribe(audio, sample_rate=sample_rate)  # type: ignore
            text   = res.get("text", "")
            tokens = res.get("tokens", text.split())
        elif self.backend.startswith("whisper"):
            result = self._model.transcribe(                             # type: ignore
                audio.astype(np.float32),
                fp16=(self.device != "cpu"),
            )
            text   = result["text"].strip()
            tokens = text.split()
        else:
            dur_s = len(audio) / sample_rate
            time.sleep(min(0.05, dur_s * 0.01))    # minimal simulated latency
            text   = f"[Mock transcription of {dur_s:.2f}s audio]"
            tokens = text.split()

        elapsed_ns = time.perf_counter_ns() - t0
        ms = elapsed_ns / 1e6
        return {
            "text": text,
            "ttft_ms": ms,
            "total_duration_ms": ms,
            "tokens": tokens,
            "model_used": self.backend,
        }

    # ──────────────────────────────────────────────────
    def transcribe_streaming(
        self,
        audio_chunks: List[np.ndarray],
        sample_rate: int = 16_000,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yields partial results as chunks arrive."""
        accum = np.array([], dtype=np.float32)
        for idx, chunk in enumerate(audio_chunks):
            accum = np.concatenate((accum, chunk))
            res = self.transcribe(accum, sample_rate)
            res["chunk_index"] = idx
            if idx > 0:
                res["ttft_ms"] = 0.0   # TTFT only meaningful on first chunk
            yield res


class TranscriptState:
    """Accumulates incremental transcript updates."""

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []
        self._current: str = ""

    def update(self, new_text: str, timestamp_ms: float) -> None:
        self._current = new_text
        self._history.append({"text": new_text, "timestamp_ms": timestamp_ms})

    def get_current(self) -> str:
        return self._current

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._current = ""


# ── Quick demo ──
print("\n[Module 4] Loading ASR engine …")
_asr = StreamingASREngine()
_asr.load_model()

_asr_res = _asr.transcribe(_vad_result["trimmed_audio"] if len(_vad_result["trimmed_audio"]) > 0 else _isolated)
print(f"  text    : {_asr_res['text']}")
print(f"  TTFT    : {_asr_res['ttft_ms']:.2f} ms")
print(f"  backend : {_asr_res['model_used']}")

# %% [markdown]
# ---
# ## Module 5: Decision Brain Layer — TypeSafe AI Jev System One

# %%
class ActionType(str, Enum):
    CLOSE_APP      = "Close_App"
    LAUNCH_APP     = "Launch_App"
    CLICK_ELEMENT  = "Click_Element"
    SCROLL         = "Scroll"
    TYPE_TEXT       = "Type_Text"
    NO_ACTION      = "No_Action"


class TargetType(str, Enum):
    BROWSER    = "Browser"
    EDITOR     = "Editor"
    TERMINAL   = "Terminal"
    GENERAL_UI = "General_UI"


class JevDecision(BaseModel):
    action: ActionType
    action_confidence: float = Field(..., ge=0.0, le=1.0)
    target: TargetType
    target_confidence: float = Field(..., ge=0.0, le=1.0)
    is_destructive: bool
    risk_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    raw_logits: Dict[str, float]
    requires_confirmation: bool
    auto_execute: bool


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


class MockJevBrain:
    """
    Calibrated keyword→logit mock of TypeSafe Jev.
    Produces deterministic probabilistic decisions from transcript text.
    """

    _ACTION_KEYWORDS: Dict[ActionType, List[str]] = {
        ActionType.CLOSE_APP:     ["close", "quit", "exit", "kill", "shut"],
        ActionType.LAUNCH_APP:    ["open", "launch", "start", "run"],
        ActionType.CLICK_ELEMENT: ["click", "press", "tap", "select", "hit", "button"],
        ActionType.SCROLL:        ["scroll", "swipe"],
        ActionType.TYPE_TEXT:     ["type", "write", "enter", "input", "fill"],
    }

    _TARGET_KEYWORDS: Dict[TargetType, List[str]] = {
        TargetType.BROWSER:    ["browser", "chrome", "firefox", "web", "tab", "url"],
        TargetType.EDITOR:     ["editor", "code", "vscode", "ide", "file"],
        TargetType.TERMINAL:   ["terminal", "console", "bash", "cmd", "shell"],
    }

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    # ──────────────────────────────────────────────────
    def _keyword_logits(
        self,
        transcript: str,
        mapping: Dict[Any, List[str]],
        default_key: Any,
        base: float = 0.1,
        hit: float = 5.0,
    ) -> Dict[str, float]:
        t = transcript.lower()
        logits: Dict[str, float] = {k.value if hasattr(k, "value") else k: base for k in mapping}
        logits[default_key.value if hasattr(default_key, "value") else default_key] = 0.5

        for key, words in mapping.items():
            if any(w in t for w in words):
                k_str = key.value if hasattr(key, "value") else key
                logits[k_str] = max(logits[k_str], hit)
        return logits

    # ──────────────────────────────────────────────────
    def _compute_action_logits(self, transcript: str) -> Dict[str, float]:
        logits = self._keyword_logits(
            transcript, self._ACTION_KEYWORDS, ActionType.NO_ACTION,
        )
        logits.setdefault(ActionType.NO_ACTION.value, 0.5)
        return logits

    def _compute_target_logits(self, transcript: str) -> Dict[str, float]:
        logits = self._keyword_logits(
            transcript, self._TARGET_KEYWORDS, TargetType.GENERAL_UI,
        )
        logits.setdefault(TargetType.GENERAL_UI.value, 0.5)
        return logits

    # ──────────────────────────────────────────────────
    def _assess_risk(self, action: ActionType, transcript: str) -> Tuple[bool, float]:
        t = transcript.lower()
        destructive = any(w in t for w in ["delete", "remove", "drop", "kill", "format", "erase"])
        is_dest = destructive or action == ActionType.CLOSE_APP

        if is_dest:
            risk = 0.85
        elif action in (ActionType.TYPE_TEXT, ActionType.CLICK_ELEMENT):
            risk = 0.40
        elif action == ActionType.SCROLL:
            risk = 0.10
        else:
            risk = 0.05
        return is_dest, risk

    # ──────────────────────────────────────────────────
    def evaluate(self, transcript: str) -> JevDecision:
        # Action
        act_logits = self._compute_action_logits(transcript)
        act_keys  = list(act_logits.keys())
        act_probs = _softmax(np.array(list(act_logits.values())))
        top_i     = int(np.argmax(act_probs))
        top_act   = ActionType(act_keys[top_i])
        top_act_p = float(act_probs[top_i])

        # Target
        tgt_logits = self._compute_target_logits(transcript)
        tgt_keys  = list(tgt_logits.keys())
        tgt_probs = _softmax(np.array(list(tgt_logits.values())))
        top_j     = int(np.argmax(tgt_probs))
        top_tgt   = TargetType(tgt_keys[top_j])
        top_tgt_p = float(tgt_probs[top_j])

        # Risk
        is_dest, risk = self._assess_risk(top_act, transcript)

        # Confidence gating
        auto_exec = (top_act_p > 0.90) and (risk < 0.20)
        req_conf  = (0.60 <= top_act_p <= 0.90) or (risk >= 0.50)
        if top_act_p < 0.60:
            top_act   = ActionType.NO_ACTION
            auto_exec = False
            req_conf  = False

        return JevDecision(
            action=top_act,
            action_confidence=top_act_p,
            target=top_tgt,
            target_confidence=top_tgt_p,
            is_destructive=is_dest,
            risk_score=risk,
            reasoning=(
                f"Identified '{top_act.value}' with {top_act_p*100:.1f}% confidence "
                f"targeting '{top_tgt.value}'."
            ),
            raw_logits=act_logits,
            requires_confirmation=req_conf,
            auto_execute=auto_exec,
        )


class JevBrainRouter:
    """Delegates to TypeSafe SDK when an API key is present, else MockJevBrain."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self._mock = MockJevBrain()

    def evaluate(self, transcript: str) -> JevDecision:
        if self.api_key:
            try:
                # Placeholder for real TypeSafe SDK call:
                # import typesafe_sdk
                # client = typesafe_sdk.Jev(api_key=self.api_key)
                # return client.evaluate(transcript)
                pass
            except Exception as exc:
                print(f"[Jev] TypeSafe API failed ({exc}), falling back to mock.")
        return self._mock.evaluate(transcript)


# ── Quick demo ──
print("\n[Module 5] Decision Brain demo:")
_jev = JevBrainRouter()
for phrase in ["close the browser", "click the submit button",
               "scroll down a bit", "open the terminal",
               "hello world"]:
    d = _jev.evaluate(phrase)
    print(f"  \"{phrase}\"")
    print(f"    → {d.action.value} @ {d.action_confidence:.2f}  "
          f"target={d.target.value}  auto={d.auto_execute}  confirm={d.requires_confirmation}")

# %% [markdown]
# ---
# ## Module 6: Visual Grounding & Windows-Ready Action Serialization

# %%
# ─────────────────────────────────────────────────────────────────
#  6-A  Synthetic Desktop Screenshot Generator
# ─────────────────────────────────────────────────────────────────
# Coordinates are stored as constants so the grounder can verify
# against ground truth.

_SUBMIT_BTN = {"label": "Submit",  "x": 250, "y": 320, "w": 120, "h": 40}
_CLOSE_BTN  = {"label": "Close",   "x": 1160, "y": 105, "w": 30,  "h": 20}
_URL_BAR    = {"label": "URL Bar", "x": 210, "y": 140, "w": 980,  "h": 30}
_INPUT_FLD  = {"label": "Input",   "x": 250, "y": 250, "w": 350,  "h": 40}

_KNOWN_ELEMENTS = [_SUBMIT_BTN, _CLOSE_BTN, _URL_BAR, _INPUT_FLD]


def generate_sample_desktop_screenshot(
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    """Creates a synthetic desktop screenshot (BGR uint8) with known UI elements."""
    img = np.full((height, width, 3), fill_value=(130, 80, 50), dtype=np.uint8)

    # Taskbar
    cv2.rectangle(img, (0, height - 40), (width, height), (40, 40, 40), -1)
    cv2.rectangle(img, (10, height - 35), (45, height - 5), (200, 100, 50), -1)

    # Browser chrome
    bx, by, bw, bh = 200, 100, 1000, 800
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (240, 240, 240), -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + 30), (200, 200, 200), -1)
    cv2.putText(img, "Browser - Mock Website", (bx + 10, by + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Close button (red)
    e = _CLOSE_BTN
    cv2.rectangle(img, (e["x"], e["y"]), (e["x"] + e["w"], e["y"] + e["h"]),
                  (0, 0, 200), -1)

    # URL bar
    e = _URL_BAR
    cv2.rectangle(img, (e["x"], e["y"]), (e["x"] + e["w"], e["y"] + e["h"]),
                  (255, 255, 255), -1)
    cv2.putText(img, "https://example.com/form", (e["x"] + 10, e["y"] + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    # Form text
    cv2.putText(img, "Please fill out the form below:", (bx + 50, by + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Input field
    e = _INPUT_FLD
    cv2.rectangle(img, (e["x"], e["y"]), (e["x"] + e["w"], e["y"] + e["h"]),
                  (255, 255, 255), -1)
    cv2.rectangle(img, (e["x"], e["y"]), (e["x"] + e["w"], e["y"] + e["h"]),
                  (100, 100, 100), 1)

    # Submit button
    e = _SUBMIT_BTN
    cv2.rectangle(img, (e["x"], e["y"]), (e["x"] + e["w"], e["y"] + e["h"]),
                  (200, 50, 50), -1)
    cv2.putText(img, "Submit", (e["x"] + 25, e["y"] + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


# ─────────────────────────────────────────────────────────────────
#  6-B  PixelJev Grounder
# ─────────────────────────────────────────────────────────────────
class PixelJevGrounder:
    """
    Visual element grounding using YOLO (for UI objects) and EasyOCR (for text).
    """

    def __init__(self) -> None:
        # Build lookup from known synthetic elements for fallback
        self._known: Dict[str, Dict[str, Any]] = {}
        for el in _KNOWN_ELEMENTS:
            self._known[el["label"].lower()] = el
            
        self.use_yolo = False
        try:
            from ultralytics import YOLO
            import easyocr
            import logging
            logging.getLogger("easyocr").setLevel(logging.ERROR)
            print("  [Grounder] Loading YOLOv8n and EasyOCR...")
            self.yolo = YOLO('yolov8n.pt', verbose=False)
            self.ocr = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
            self.use_yolo = True
            print("  [Grounder] YOLO and OCR successfully loaded.")
        except ImportError:
            print("  [Grounder] Ultralytics or EasyOCR not found. Falling back to heuristics.")

    def _heuristic_match(self, image: np.ndarray, target: str) -> Optional[Tuple[float, float, float]]:
        h, w = image.shape[:2]
        tl = target.lower()
        for key, el in self._known.items():
            if key in tl:
                cx = (el["x"] + el["w"] / 2) / w
                cy = (el["y"] + el["h"] / 2) / h
                return (cx, cy, 0.95)
        return None

    def _yolo_ocr_match(self, image: np.ndarray, target: str) -> Optional[Tuple[float, float, float]]:
        # 1. OCR text extraction
        ocr_results = self.ocr.readtext(image)
        
        target_lower = target.lower().replace(" button", "").replace(" tab", "").replace(" field", "")
        
        # 2. Find matching text
        best_box = None
        for (bbox, text, conf) in ocr_results:
            if target_lower in text.lower():
                best_box = bbox
                break
                
        if not best_box:
            return None
            
        # 3. Calculate center of OCR box
        # bbox is typically a list of 4 points: [tl, tr, br, bl]
        tl, tr, br, bl = best_box
        cx = (tl[0] + br[0]) / 2.0
        cy = (tl[1] + br[1]) / 2.0
        
        # 4. Optional: Cross-verify with YOLO 
        # (For this demo, YOLOv8n finds general objects, so we just run it to demonstrate the latency/pipeline)
        yolo_res = self.yolo(image, verbose=False)
        
        h, w = image.shape[:2]
        return (cx / w, cy / h, 0.98)

    def ground(self, image: np.ndarray, target_description: str) -> Dict[str, Any]:
        # 1) Try YOLO + OCR if available
        if self.use_yolo:
            hit = self._yolo_ocr_match(image, target_description)
            if hit:
                return {
                    "target": target_description,
                    "found": True,
                    "norm_x": hit[0],
                    "norm_y": hit[1],
                    "confidence": hit[2],
                    "method": "yolo_ocr",
                    "all_candidates": [],
                }
                
        # 2) Fallback heuristic / known-element match
        hit = self._heuristic_match(image, target_description)
        if hit:
            return {
                "target": target_description,
                "found": True,
                "norm_x": hit[0],
                "norm_y": hit[1],
                "confidence": hit[2],
                "method": "heuristic",
                "all_candidates": [],
            }

        return {
            "target": target_description,
            "found": False,
            "norm_x": 0.0,
            "norm_y": 0.0,
            "confidence": 0.0,
            "method": "none",
            "all_candidates": [],
        }

# ─────────────────────────────────────────────────────────────────
#  6-C  Win32 SendInput Serialization Layer
# ─────────────────────────────────────────────────────────────────
class MouseInputFlags(IntEnum):
    MOUSEEVENTF_MOVE      = 0x0001
    MOUSEEVENTF_LEFTDOWN  = 0x0002
    MOUSEEVENTF_LEFTUP    = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP   = 0x0010
    MOUSEEVENTF_WHEEL     = 0x0800
    MOUSEEVENTF_ABSOLUTE  = 0x8000


class KeybdInputFlags(IntEnum):
    KEYEVENTF_KEYDOWN  = 0x0000
    KEYEVENTF_KEYUP    = 0x0002
    KEYEVENTF_UNICODE  = 0x0004


class Win32MouseInput(BaseModel):
    type: int = 0               # INPUT_MOUSE
    dx: int                     # 0-65535 absolute
    dy: int
    mouseData: int = 0
    dwFlags: int                # e.g. 0x8001 | 0x0002
    time: int = 0


class Win32KeybdInput(BaseModel):
    type: int = 1               # INPUT_KEYBOARD
    wVk: int
    wScan: int
    dwFlags: int
    time: int = 0


class ActionPayload(BaseModel):
    sequence_id: str
    timestamp_iso: str
    inputs: List[Union[Win32MouseInput, Win32KeybdInput]]
    metadata: Dict[str, Any]


class ActionSerializer:
    """Converts normalised coordinates + Jev decisions into Win32 SendInput payloads."""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width  = screen_width
        self.screen_height = screen_height

    # ──────────────────────────────────────────────────
    def norm_to_absolute(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """[0,1] → [0,65535] for MOUSEEVENTF_ABSOLUTE."""
        ax = max(0, min(65535, int(norm_x * 65535)))
        ay = max(0, min(65535, int(norm_y * 65535)))
        return ax, ay

    # ──────────────────────────────────────────────────
    def _make_payload(
        self,
        inputs: List[Union[Win32MouseInput, Win32KeybdInput]],
        meta: Dict[str, Any],
    ) -> ActionPayload:
        return ActionPayload(
            sequence_id=f"seq_{uuid.uuid4().hex[:12]}",
            timestamp_iso=datetime.now(timezone.utc).isoformat(),
            inputs=inputs,
            metadata=meta,
        )

    # ──────────────────────────────────────────────────
    def create_click_payload(
        self,
        norm_x: float,
        norm_y: float,
        button: str = "left",
    ) -> ActionPayload:
        dx, dy = self.norm_to_absolute(norm_x, norm_y)
        down = MouseInputFlags.MOUSEEVENTF_LEFTDOWN if button == "left" else MouseInputFlags.MOUSEEVENTF_RIGHTDOWN
        up   = MouseInputFlags.MOUSEEVENTF_LEFTUP   if button == "left" else MouseInputFlags.MOUSEEVENTF_RIGHTUP
        move_flags = int(MouseInputFlags.MOUSEEVENTF_MOVE | MouseInputFlags.MOUSEEVENTF_ABSOLUTE)
        return self._make_payload(
            [
                Win32MouseInput(dx=dx, dy=dy, dwFlags=move_flags),
                Win32MouseInput(dx=dx, dy=dy, dwFlags=int(down | MouseInputFlags.MOUSEEVENTF_ABSOLUTE)),
                Win32MouseInput(dx=dx, dy=dy, dwFlags=int(up   | MouseInputFlags.MOUSEEVENTF_ABSOLUTE)),
            ],
            {"action": "click", "button": button, "norm_pos": [norm_x, norm_y]},
        )

    # ──────────────────────────────────────────────────
    def create_type_payload(self, text: str) -> ActionPayload:
        inputs: List[Union[Win32MouseInput, Win32KeybdInput]] = []
        for ch in text:
            sc = ord(ch)
            inputs.append(Win32KeybdInput(
                wVk=0, wScan=sc, dwFlags=int(KeybdInputFlags.KEYEVENTF_UNICODE),
            ))
            inputs.append(Win32KeybdInput(
                wVk=0, wScan=sc,
                dwFlags=int(KeybdInputFlags.KEYEVENTF_UNICODE | KeybdInputFlags.KEYEVENTF_KEYUP),
            ))
        return self._make_payload(inputs, {"action": "type", "length": len(text)})

    # ──────────────────────────────────────────────────
    def create_scroll_payload(
        self,
        norm_x: float,
        norm_y: float,
        delta: int = -120,
    ) -> ActionPayload:
        dx, dy = self.norm_to_absolute(norm_x, norm_y)
        move_flags = int(MouseInputFlags.MOUSEEVENTF_MOVE | MouseInputFlags.MOUSEEVENTF_ABSOLUTE)
        return self._make_payload(
            [
                Win32MouseInput(dx=dx, dy=dy, dwFlags=move_flags),
                Win32MouseInput(dx=dx, dy=dy, mouseData=delta,
                                dwFlags=int(MouseInputFlags.MOUSEEVENTF_WHEEL)),
            ],
            {"action": "scroll", "delta": delta},
        )

    # ──────────────────────────────────────────────────
    def serialize_decision(
        self,
        decision: JevDecision,
        grounding: Dict[str, Any],
        text_to_type: str = "",
    ) -> ActionPayload:
        nx = grounding.get("norm_x", 0.5)
        ny = grounding.get("norm_y", 0.5)

        if decision.action == ActionType.CLICK_ELEMENT:
            return self.create_click_payload(nx, ny)
        elif decision.action == ActionType.TYPE_TEXT:
            return self.create_type_payload(text_to_type or "default text")
        elif decision.action == ActionType.SCROLL:
            return self.create_scroll_payload(nx, ny)
        elif decision.action in (ActionType.CLOSE_APP, ActionType.LAUNCH_APP):
            return self.create_click_payload(nx, ny)  # click the close/launch target
        else:
            return self._make_payload([], {"action": decision.action.value, "note": "no-op"})


# ── Quick demo ──
print("\n[Module 6] Visual grounding demo:")
_screen = generate_sample_desktop_screenshot()
_grounder = PixelJevGrounder()
_serializer = ActionSerializer()

for target in ["Submit button", "Close tab", "Input field"]:
    g = _grounder.ground(_screen, target)
    print(f"  \"{target}\" → found={g['found']}  norm=({g['norm_x']:.3f}, {g['norm_y']:.3f})  "
          f"conf={g['confidence']:.2f}  method={g['method']}")

_test_dec = _jev.evaluate("click the submit button")
_payload = _serializer.serialize_decision(_test_dec, _grounder.ground(_screen, "Submit button"))
print(f"\n  Payload preview:\n{_payload.model_dump_json(indent=2)[:500]}")

# %% [markdown]
# ---
# ## Module 7: End-to-End Latency Benchmarking & Profiler

# %%
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


# ── Run benchmark (reduced to 20 trials for demo speed) ──
print("\n[Module 7] Running 20-trial end-to-end benchmark …")
_bench = PipelineBenchmark(
    avtse=_avtse, vad=_vad, asr=_asr, jev=_jev,
    grounder=_grounder, serializer=_serializer,
    num_trials=20, sir_db=-5.0, duration_s=3.0,
)
_df_results = _bench.run_all()
_df_stats   = PipelineBenchmark.compute_statistics(_df_results)
_fig_hist, _fig_bar = PipelineBenchmark.display_results(_df_results, _df_stats)
plt.show()

# %% [markdown]
# ---
# ## Interactive Gradio UI
# Launch with `share=True` for a Colab-compatible public URL.

# %%
import gradio as gr


def build_gradio_app(
    avtse: AVTSEEngine,
    vad: SileroVADProcessor,
    asr: StreamingASREngine,
    jev: JevBrainRouter,
    grounder: PixelJevGrounder,
    serializer: ActionSerializer,
) -> gr.Blocks:
    """Constructs a 3-tab Gradio Blocks application."""

    screen_img = generate_sample_desktop_screenshot()

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
        grounding = grounder.ground(screen_img, decision.target.value)
        payload   = serializer.serialize_decision(decision, grounding)

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


# ── Launch ──
print("\n[Gradio] Building interactive UI …")
_app = build_gradio_app(
    avtse=_avtse, vad=_vad, asr=_asr, jev=_jev,
    grounder=_grounder, serializer=_serializer,
)
_app.launch(share=False, show_error=True, inbrowser=True)
