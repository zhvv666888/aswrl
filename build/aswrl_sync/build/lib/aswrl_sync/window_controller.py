from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

WINDOW_SET_DEFAULT = [1, 2, 4, 8, 16, 32, 64]

@dataclass
class WindowAction:
    action: int  # -1, 0, +1

class WindowSmoother:
    def __init__(self, W: int):
        self.set_W(W)

    def set_W(self, W: int):
        W = int(W)
        self.W = max(1, W)
        self.buf = deque(maxlen=self.W)

    def add(self, x: float) -> float:
        self.buf.append(float(x))
        return float(np.mean(self.buf)) if len(self.buf) else float(x)

    def var(self) -> float:
        if len(self.buf) < 2:
            return 0.0
        return float(np.var(np.array(self.buf, dtype=np.float64)))

class FixedWindowController:
    def __init__(self, fixed_window: int):
        self.fixed_window = int(fixed_window)

    def choose_W(self, *_args, **_kwargs) -> int:
        return self.fixed_window

class RuleASWController:
    """Simple rule-based ASW baseline.

    If delay variance increases -> expand window (stabilize).
    If delay variance decreases and sync error is stable -> shrink window (increase responsiveness).

    This is a practical baseline (thresholds can be tuned).
    """
    def __init__(self, window_set: Optional[List[int]] = None,
                 var_hi: float = 5e14, var_lo: float = 1e14,
                 err_hi_ns: float = 5e6, err_lo_ns: float = 1e6):
        self.window_set = window_set or WINDOW_SET_DEFAULT
        self.idx = self.window_set.index(16) if 16 in self.window_set else len(self.window_set)//2
        self.var_hi = float(var_hi)
        self.var_lo = float(var_lo)
        self.err_hi_ns = float(err_hi_ns)
        self.err_lo_ns = float(err_lo_ns)

    def choose_W(self, delay_var: float, sync_err_ns: float) -> int:
        if delay_var > self.var_hi or abs(sync_err_ns) > self.err_hi_ns:
            self.idx = min(self.idx + 1, len(self.window_set) - 1)
        elif delay_var < self.var_lo and abs(sync_err_ns) < self.err_lo_ns:
            self.idx = max(self.idx - 1, 0)
        return self.window_set[self.idx]

class RLASWController:
    """DQN policy-based window controller (inference only).

    The policy file is a torchscript or state_dict model exported by rl/export_policy.py.
    """
    def __init__(self, policy, window_set: Optional[List[int]] = None):
        self.policy = policy
        self.window_set = window_set or WINDOW_SET_DEFAULT
        self.idx = self.window_set.index(16) if 16 in self.window_set else len(self.window_set)//2

    def choose_W(self, state_vec) -> int:
        # state_vec: 1D numpy array
        import torch
        with torch.no_grad():
            x = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            q = self.policy(x)
            a = int(torch.argmax(q, dim=1).item())  # 0,1,2 -> -1,0,+1
        delta = a - 1
        self.idx = int(np.clip(self.idx + delta, 0, len(self.window_set)-1))
        return self.window_set[self.idx]
