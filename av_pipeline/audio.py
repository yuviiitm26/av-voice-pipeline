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
            import platform
            default_cache = r"D:\ai_cache\whisper" if platform.system() == "Windows" else "~/.cache/whisper"
            whisper_cache = os.environ.get("WHISPER_CACHE_DIR", os.path.expanduser(default_cache))
            self._model = whisper.load_model("tiny", device=self.device, download_root=whisper_cache)
            self.backend = "whisper-tiny"
            print(f"[ASR] Whisper-tiny loaded from {whisper_cache}.")
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


