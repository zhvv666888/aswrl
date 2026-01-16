import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

WINDOW_SET = [1,2,4,8,16,32,64]

class ASWRLEnv(gym.Env):
    """
    Improved ASW-RL Environment with physically realistic lag.
    
    Key Changes from original:
    1. Maintains a history buffer of raw measurements.
    2. Window smoothing is applied explicitly (mean of last W samples).
    3. Large windows now naturally cause LAG when offset drifts or jumps.
    4. Reward is based on TRUE error (Sim-to-Real privilege).
    """
    metadata = {"render_modes": []}

    def __init__(self, scenario: str = "multimodal", seed: int = 0, dt: float = 0.1):
        super().__init__()
        self.scenario = scenario
        self.rnd = random.Random(seed)
        self.dt = dt

        # Action: 0:down, 1:stay, 2:up
        self.action_space = spaces.Discrete(3)
        
        # Observation: [delay_mean, delay_var, loss, current_est_offset, delta_est, window_norm]
        # We increase bounds slightly to handle drift accumulation
        hi = np.array([5.0, 5.0, 1.0, 10.0, 10.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)

        # Max history matches max window size
        self.max_window = max(WINDOW_SET)
        self.raw_history = deque(maxlen=self.max_window)

        self.reset(seed=seed)

    def _net_params(self, t: float):
        # return (delay_ms, jitter_ms, loss)
        if self.scenario == "stable":
            return (20.0, 1.0, 0.001)
        if self.scenario == "jitter":
            # Jitter 25ms is significant
            return (30.0, 25.0, 0.02)
        if self.scenario == "multimodal":
            tp = t % 120.0
            if tp < 30: return (20.0, 5.0, 0.005)
            if tp < 60: return (80.0, 10.0, 0.015)
            if tp < 90: return (35.0, 20.0, 0.02)
            return (100.0, 5.0, 0.002)
        if self.scenario == "route_switch":
            tp = t % 90.0
            if tp < 20: return (25.0, 3.0, 0.002)
            if tp < 45: return (120.0, 30.0, 0.03)
            if tp < 70: return (60.0, 10.0, 0.01)
            return (25.0, 3.0, 0.002)
        return (30.0, 10.0, 0.01)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rnd.seed(seed)
        self.t = 0.0
        self.drift_ppm = self.rnd.uniform(10.0, 60.0)
        self.offset_s = self.rnd.uniform(-0.01, 0.01)
        
        self.idx = WINDOW_SET.index(16)
        self.delay_hist = []
        self.raw_history.clear()
        
        # Pre-fill history to avoid empty buffer issues at start
        # Assume perfect history initially centered at offset_s
        for _ in range(self.max_window):
            self.raw_history.append(self.offset_s)

        self.last_est = self.offset_s
        
        # Initial observation
        obs = self._obs(delay=0.02, delay_var=0.0, loss=0.0, est=self.offset_s, dest=0.0)
        return obs, {}

    def _obs(self, delay, delay_var, loss, est, dest):
        return np.array([
            float(delay),
            float(delay_var),
            float(loss),
            float(est),      # Robot sees the ESTIMATED offset
            float(dest),     # Change in estimate
            float(WINDOW_SET[self.idx]) / 64.0
        ], dtype=np.float32)

    def step(self, action: int):
        # 1. Update Window Index
        delta = int(action) - 1
        self.idx = max(0, min(len(WINDOW_SET)-1, self.idx + delta))
        W = WINDOW_SET[self.idx]

        # 2. Simulate Network Physics
        delay_ms, jitter_ms, loss = self._net_params(self.t)
        
        # One-way delay simulation
        delay = max(0.0, self.rnd.gauss(delay_ms, jitter_ms)) / 1000.0
        if self.rnd.random() < loss:
            delay += self.rnd.uniform(0.05, 0.2) # Loss penalty

        # Network stats for observation (features)
        self.delay_hist.append(delay)
        if len(self.delay_hist) > 64: # Track stats over max window
            self.delay_hist = self.delay_hist[-64:]
        delay_mean = float(np.mean(self.delay_hist))
        delay_var = float(np.var(self.delay_hist)) if len(self.delay_hist) > 1 else 0.0

        # 3. Evolve True Clock Physics
        drift = self.drift_ppm * 1e-6
        self.offset_s += drift * self.dt # True offset drifts over time

        # 4. Generate RAW Noisy Measurement (No smoothing yet!)
        # Noise comes purely from jitter (and other unmodeled noise)
        # In PTP, meas noise ~ jitter / 2 roughly, but here we simplify
        raw_noise_std = (jitter_ms / 1000.0) 
        raw_meas = self.offset_s + self.rnd.gauss(0.0, raw_noise_std)
        
        self.raw_history.append(raw_meas)

        # 5. Apply Window Smoothing (The Agent's Action Effect)
        # Take the last W samples
        current_samples = list(self.raw_history)[-W:]
        est_offset = float(np.mean(current_samples))

        # Calculate dynamics for observation
        dest_offset = est_offset - self.last_est
        self.last_est = est_offset

        # 6. Calculate Reward based on TRUE ERROR (Privileged info)
        # Ideally, we want est_offset == offset_s
        # BUT, if W is huge and offset_s is drifting, est_offset will lag behind!
        true_error = est_offset - self.offset_s
        
        # Reward function:
        # - Penalize square of true error (MSE proxy)
        # - Penalize window size slightly (computational cost / response ability)
        # - Penalize huge oscillations in estimate
        r = - (abs(true_error) * 10.0 + abs(dest_offset) * 2.0 + (W / 64.0) * 0.1)

        self.t += self.dt
        terminated = False
        truncated = self.t > 120.0
        
        # 7. Observation
        # Note: We feed 'est_offset' to the agent, not the raw error. 
        # The agent sees "where I am" and "how jittery the network is".
        obs = self._obs(delay_mean, delay_var, loss, est_offset, dest_offset)
        
        info = {
            "W": W, 
            "true_error": true_error, # For logging
            "raw_meas": raw_meas,
            "lag": true_error,        # Lag is essentially the error
            "jitter_ms": jitter_ms
        }
        return obs, float(r), terminated, truncated, info