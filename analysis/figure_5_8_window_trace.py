#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5.8 - Window Length Trace (Final Fixed Version)
======================================================
Scenario: Jitter
Index 3 is confirmed as Window Size.
Behavior: FW=16, Rule=64 (Saturated), RL=1 (Minimizing lag).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from rosbag2_sqlite_utils import list_topics, load_multiarray_index_series

# ----------------------------
# USER CONFIG
# ----------------------------
SCENARIO_NAME = "Jitter"

# 你的路径
BAG_FW   = "/home/zhvv/aswrl_bags/jitter_fw_001/jitter_fw_001_0.db3"
BAG_RULE = "/home/zhvv/aswrl_bags/jitter_rule_001/jitter_rule_001_0.db3"
BAG_RL   = "/home/zhvv/aswrl_bags/jitter_aswrl_001/jitter_aswrl_001_0.db3"

# 经排查确认：Index 3 是窗口大小
STATUS_TOPIC = "/sync/status"
WINDOW_INDEX = 3  
WARMUP_SEC = 0.0

OUT_PNG = "fig_5_8_window_trace_jitter.png"
DOWNSAMPLE_STEP = 10  # 数据较多，稍微降采样加速绘图

# ----------------------------
# Helpers
# ----------------------------
def load_window_series(db_path: str, label: str):
    print(f"[{label}] Loading {STATUS_TOPIC} index {WINDOW_INDEX} from {db_path}...")
    try:
        t, y, _ = load_multiarray_index_series(
            db_path=db_path,
            topic=STATUS_TOPIC,
            index=WINDOW_INDEX,
            warmup_sec=WARMUP_SEC,
            unit_scale=1.0,
            fuzzy_topic=False,
        )
        if len(t) == 0:
            print(f"  [WARN] No data found for {label}!")
            return np.array([]), np.array([])
        t = t - t[0]
        return t, y
    except Exception as e:
        print(f"  [ERROR] Failed to load {label}: {e}")
        return np.array([]), np.array([])

def trim_to_common_end(series_dict):
    ends = [t[-1] for t, _ in series_dict.values() if len(t) > 0]
    if not ends: return series_dict
    common_end = min(ends)
    return {k: (t[t<=common_end], y[t<=common_end]) for k, (t,y) in series_dict.items()}

# ----------------------------
# Main
# ----------------------------
def main():
    # 1. Load Data
    data = {}
    data["FW"]       = load_window_series(BAG_FW, "FW")
    data["Rule-ASW"] = load_window_series(BAG_RULE, "Rule-ASW")
    data["ASW-RL"]   = load_window_series(BAG_RL, "ASW-RL")

    # 2. Trim & Downsample
    data = trim_to_common_end(data)
    
    # 3. Plot
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = plt.gca()
    
    colors = {"FW": "tab:blue", "Rule-ASW": "tab:orange", "ASW-RL": "tab:green"}
    linestyles = {"FW": "--", "Rule-ASW": "-", "ASW-RL": "-"}

    for name in ["FW", "Rule-ASW", "ASW-RL"]:
        t, y = data[name]
        if len(t) == 0: continue
        
        # 下采样
        t_ds = t[::DOWNSAMPLE_STEP]
        y_ds = y[::DOWNSAMPLE_STEP]
        
        # 绘制
        ax.plot(t_ds, y_ds, label=name, color=colors[name], 
                linestyle=linestyles[name], linewidth=2.5, alpha=0.9)

        # 打印统计，放入图注或控制台
        print(f"  -> {name}: Mean={np.mean(y):.1f}, Std={np.std(y):.3f}")

    # 4. Formatting
    ax.set_title(f"Window Length Trace under {SCENARIO_NAME} Scenario", fontsize=14)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Window Length (count)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12, loc='center right')
    
    # 设置 Y 轴范围，让三条线都看得清 (0 到 70)
    ax.set_ylim(-5, 75)
    
    # 添加文字说明，解释为什么是直的 (可选)
    # ax.text(0.5, 0.05, "Note: Rule-ASW saturates at max (64); ASW-RL converges to min (1)", 
    #         transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print(f"\n[OK] Saved plot to {OUT_PNG}")

if __name__ == "__main__":
    main()
