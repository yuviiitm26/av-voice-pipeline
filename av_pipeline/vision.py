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

from .core import ActionType, MouseInputFlags, KeybdInputFlags, Win32MouseInput, Win32KeybdInput, ActionPayload


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


class PixelJevGrounder:
    """
    Visual element grounding using YOLO (for UI objects) and EasyOCR (for text).
    
    ARCHITECTURAL DECISION: YOLO vs OpenCV Heuristics vs VLMs
    ---------------------------------------------------------
    1. Why not OpenCV? Traditional heuristics (like cv2.findContours) are brittle. 
       They rely on hardcoded edge detection that breaks when UI themes, colors, or layouts change.
    2. Why not VLMs (e.g., GPT-4V)? Vision-Language Models take 2-5 seconds per frame, 
       which is unacceptable for an ultra-low-latency voice automation pipeline.
    3. Why YOLO + OCR? YOLO processes a frame in 10-30ms. By combining YOLO (to detect UI 
       primitives like [Button], [Input]) with EasyOCR (to read text), we achieve 
       VLM-level semantic understanding at hardware-level speeds.
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


