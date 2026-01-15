#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5.9 - Statistical Significance Analysis (Log Scale Version)
==================================================================
- Scenarios: Stable + Jitter (Multimodal skipped automatically if empty)
- Y-Axis: Logarithmic (to handle Rule-ASW's huge divergence error)
- Data: /sync/status [Index 2] (Offset in us -> ms)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from pathlib import Path
from rosbag2_sqlite_utils import load_multiarray_index_series

# ----------------------------
# USER CONFIG
# ----------------------------
SCENARIOS = {
    "Stable": {
        "FW":   "/home/zhvv/aswrl_bags/stable_fw_001/stable_fw_001_0.db3",
        "Rule": "/home/zhvv/aswrl_bags/stable_rule_001/stable_rule_001_0.db3",
        "RL":   "/home/zhvv/aswrl_bags/stable_aswrl_001/stable_aswrl_001_0.db3",
    },
    "Jitter": {
        "FW":   "/home/zhvv/aswrl_bags/jitter_fw_001/jitter_fw_001_0.db3",
        "Rule": "/home/zhvv/aswrl_bags/jitter_rule_001/jitter_rule_001_0.db3",
        "RL":   "/home/zhvv/aswrl_bags/jitter_aswrl_001/jitter_aswrl_001_0.db3",
    },
    # Multimodal 被注释掉，因为数据为空
    # "Multimodal": { ... } 
}

# 只能用 /sync/status Index 2
ERROR_TOPIC = "/sync/status"
ERROR_INDEX = 2
WARMUP_SEC = 2.0 

METRIC_NAME = "MAE (ms)"
OUT_PNG = "fig_5_9_significance_log.png"
OUT_CSV = "table_5_9_stats.csv"

# ----------------------------
# Helpers
# ----------------------------
def get_error_series(db_path, sc_name, method_name):
    if not db_path or not Path(db_path).exists():
        return None
    try:
        _, y, _ = load_multiarray_index_series(
            db_path, ERROR_TOPIC, ERROR_INDEX, 
            warmup_sec=WARMUP_SEC, fuzzy_topic=False
        )
        if len(y) == 0:
            print(f"[WARN] {sc_name}-{method_name}: No data found!")
            return None
            
        # 绝对值 + 微秒转毫秒
        return np.abs(y) / 1000.0 
    except Exception:
        return None

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def run_stats(x, y):
    # Mann-Whitney U test & Cohen's d
    try:
        _, p_val = stats.mannwhitneyu(x, y, alternative='two-sided')
    except: p_val = 1.0
    
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    d = (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 1e-9 else 0.0
    
    return p_val, d

# ----------------------------
# Main
# ----------------------------
def main():
    results = []
    
    # 画布
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    bar_width = 0.25
    
    plot_means = {"FW": [], "Rule": [], "RL": []}
    plot_cis   = {"FW": [], "Rule": [], "RL": []}
    valid_scenarios = []

    for sc_name, paths in SCENARIOS.items():
        print(f"\n--- {sc_name} ---")
        
        # 加载数据
        e_fw = get_error_series(paths["FW"], sc_name, "FW")
        e_rule = get_error_series(paths["Rule"], sc_name, "Rule")
        e_rl = get_error_series(paths["RL"], sc_name, "RL")
        
        if e_fw is None or e_rule is None or e_rl is None:
            print(f"Skipping {sc_name} (Missing Data)")
            continue
            
        valid_scenarios.append(sc_name)
        
        # 统计
        series_map = {"FW": e_fw, "Rule": e_rule, "RL": e_rl}
        for m_name in ["FW", "Rule", "RL"]:
            d = series_map[m_name]
            mean, ci = mean_confidence_interval(d)
            plot_means[m_name].append(mean)
            plot_cis[m_name].append(ci)
            
            # 记录到表格
            res = {
                "Scenario": sc_name, "Method": m_name,
                "MAE": mean, "CI": ci, "Max": np.max(d)
            }
            if m_name == "RL":
                # 对比
                p_r, d_r = run_stats(e_rule, e_rl) # Rule vs RL
                p_f, d_f = run_stats(e_fw, e_rl)   # FW vs RL
                res.update({"p_vs_Rule": p_r, "d_vs_Rule": d_r, 
                            "p_vs_FW": p_f,   "d_vs_FW": d_f})
            
            results.append(res)
            print(f"  [{m_name}] MAE={mean:.2f} ms")

    if not valid_scenarios:
        print("No valid data to plot.")
        return

    # 画图
    x = np.arange(len(valid_scenarios))
    colors = {"FW": "tab:blue", "Rule": "tab:orange", "RL": "tab:green"}
    labels = {"FW": "FW", "Rule": "Rule-ASW", "RL": "ASW-RL"}
    
    for i, m in enumerate(["FW", "Rule", "RL"]):
        offset = (i - 1) * bar_width
        ax.bar(x + offset, plot_means[m], width=bar_width, yerr=plot_cis[m], 
               capsize=4, label=labels[m], color=colors[m], alpha=0.9, edgecolor='k')

    # 关键设置：对数坐标
    ax.set_yscale('log')
    ax.set_ylabel("MAE (ms) - Log Scale", fontsize=12, fontweight='bold')
    
    ax.set_title("Synchronization Error Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_scenarios, fontsize=12)
    ax.legend()
    ax.grid(axis='y', which="both", linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Saved {OUT_PNG} (Log Scale)")

if __name__ == "__main__":
    main()
