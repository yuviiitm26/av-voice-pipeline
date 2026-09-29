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


