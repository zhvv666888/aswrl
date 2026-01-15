import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WINDOW_SET = [1,2,4,8,16,32,64]

class ASWRLEnv(gym.Env):
    """A lightweight training environment for adaptive window selection.

    This env is intentionally simple to keep reproduction easy:
    - True clock drift is simulated (ppm)
    - Network delay varies per scenario (stable/jitter/multimodal/route_switch)
    - Agent selects window adjustment action: {0:down,1:stay,2:up}
    - Reward encourages low sync error and avoids excessively large window.

    Observation (6 floats):
      [delay_mean_s, delay_var_s2, loss_rate, sync_err_s, delta_sync_err_s, window_norm]
    """
    metadata = {"render_modes": []}

    def __init__(self, scenario: str = "multimodal", seed: int = 0, dt: float = 0.1):
        super().__init__()
        self.scenario = scenario
        self.rnd = random.Random(seed)
        self.dt = dt

        self.action_space = spaces.Discrete(3)
        hi = np.array([5, 5, 1, 5, 5, 2], dtype=np.float32)
        self.observation_space = spaces.Box(-hi, hi, dtype=np.float32)

        self.reset(seed=seed)

    def _net_params(self, t: float):
        # return (delay_ms, jitter_ms, loss)
        if self.scenario == "stable":
            return (20.0, 1.0, 0.001)
        if self.scenario == "jitter":
            return (30.0, 25.0, 0.02)
        if self.scenario == "multimodal":
            # 0-30: low, 30-60: high, 60-90: mid, 90-120: very high
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
        self.offset_s = self.rnd.uniform(-0.01, 0.01)  # initial offset in seconds
        self.last_err = 0.0
        self.idx = WINDOW_SET.index(16)
        self.delay_hist = []
        obs = self._obs(delay=0.02, delay_var=0.0, loss=0.0, err=self.offset_s, derr=0.0)
        return obs, {}

    def _obs(self, delay, delay_var, loss, err, derr):
        return np.array([
            float(delay),
            float(delay_var),
            float(loss),
            float(err),
            float(derr),
            float(WINDOW_SET[self.idx]) / 64.0
        ], dtype=np.float32)

    def step(self, action: int):
        # apply action to window index
        delta = int(action) - 1
        self.idx = max(0, min(len(WINDOW_SET)-1, self.idx + delta))
        W = WINDOW_SET[self.idx]

        # simulate network for this step
        delay_ms, jitter_ms, loss = self._net_params(self.t)
        delay = max(0.0, self.rnd.gauss(delay_ms, jitter_ms)) / 1000.0
        # drop packets => larger measurement noise; emulate by increasing jitter
        if self.rnd.random() < loss:
            delay = delay + self.rnd.uniform(0.05, 0.2)

        self.delay_hist.append(delay)
        if len(self.delay_hist) > W:
            self.delay_hist = self.delay_hist[-W:]
        delay_mean = float(np.mean(self.delay_hist))
        delay_var = float(np.var(self.delay_hist)) if len(self.delay_hist) > 1 else 0.0

        # drift evolves offset: offset += drift * dt
        drift = self.drift_ppm * 1e-6
        self.offset_s += drift * self.dt

        # measurement noise decreases with window (bigger window smoother but less responsive)
        meas_noise = (jitter_ms / 1000.0) / math.sqrt(max(1, W))
        meas = self.offset_s + self.rnd.gauss(0.0, meas_noise)

        # a simple "filter": error estimate is smoothed measurement
        # bigger window -> slower response => add lag penalty under abrupt change
        err = meas

        derr = err - self.last_err
        self.last_err = err

        # reward: small error, stable error, avoid huge window
        r = - (abs(err) * 5.0 + abs(derr) * 1.0 + (W / 64.0) * 0.2 + delay_var * 2.0)
        terminated = False
        self.t += self.dt
        truncated = self.t > 120.0  # episode length
        obs = self._obs(delay_mean, delay_var, loss, err, derr)
        info = {"W": W, "delay_ms": delay_ms, "jitter_ms": jitter_ms, "loss": loss}
        return obs, float(r), terminated, truncated, info
