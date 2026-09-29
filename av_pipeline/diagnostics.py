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


