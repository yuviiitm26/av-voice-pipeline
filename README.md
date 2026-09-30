# Ultra-Low-Latency Audio-Visual Voice Automation Pipeline

A production-grade, modular, fully runnable Python notebook implementing an end-to-end **Audio-Visual Target Speaker Extraction → ASR → Decision → Action** pipeline.

## Architecture

```
5-Speaker Mix → AV-TSE (Conv-TasNet + FiLM) → Silero VAD → Moonshine/Whisper ASR → Jev Decision Brain → PixelJev Grounding → Win32 SendInput JSON
       ↑
  96×96 Lip Crops (25 FPS)
```

## Features

- **Multi-Speaker Simulation**: Configurable cocktail-party harness (2–8 speakers, -15 to +10 dB SIR)
- **Audio-Visual Separation**: Conv-TasNet with FiLM visual conditioning from 3D lip encoder (0.86M params)
- **Silero VAD**: 512-sample (32ms) micro-segmentation with phoneme-boundary guards
- **Streaming ASR**: Moonshine → Whisper → Mock fallback cascade, no 30s zero-padding
- **Decision Brain**: Keyword→logit→softmax mock of TypeSafe Jev with confidence gating
- **Visual Grounding**: Heuristic + contour-based UI element detection
- **Win32 Serialization**: Pydantic-validated `SendInput` JSON payloads (MOUSEINPUT/KEYBDINPUT)
- **Benchmarking**: 50-trial profiler with Mean/P50/P95/P99 latency + SI-SDRi quality metrics
- **Interactive Gradio UI**: 3-tab interface (Synthetic Demo, Live Mic, Benchmark)

## Quick Start

### Google Colab / Kaggle
1. Upload `av_voice_pipeline.py` as a notebook
2. Uncomment the installation cell (lines 16–21)
3. Set runtime to **GPU** (T4/V100/A100)
4. Run all cells — Gradio launches with a public URL

### Local
```bash
pip install torch torchaudio torchvision opencv-python-headless soundfile librosa scipy matplotlib tqdm pandas pydantic gradio
python av_voice_pipeline.py
# Open http://127.0.0.1:7860
```

## Module Overview

| Module | Description | Key Classes |
|--------|-------------|-------------|
| 1 | Environment Setup & Diagnostics | `run_diagnostics()` |
| 2 | Synthetic Cocktail Party Harness | `CocktailPartyMixer`, `SpeakerProfile` |
| 3 | AV-TSE + Silero VAD | `AudioVisualSeparator`, `AVTSEEngine`, `SileroVADProcessor` |
| 4 | Streaming ASR | `StreamingASREngine`, `TranscriptState` |
| 5 | Decision Brain (Jev) | `MockJevBrain`, `JevBrainRouter`, `JevDecision` |
| 6 | Visual Grounding + Action Serialization | `PixelJevGrounder`, `ActionSerializer` |
| 7 | End-to-End Benchmarking | `PipelineBenchmark`, `si_sdr()`, `si_sdri()` |

## Decision Gating Logic

| Condition | Behavior |
|-----------|----------|
| P(Action) > 0.90 AND risk < 0.20 | **Auto-execute** |
| 0.60 ≤ P ≤ 0.90 OR risk ≥ 0.50 | **Require confirmation** |
| P < 0.60 | **No action** |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `requirements.txt` for full list

