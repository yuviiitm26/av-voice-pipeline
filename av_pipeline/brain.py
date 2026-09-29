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

from .core import ActionType, TargetType, JevDecision

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


