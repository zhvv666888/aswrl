import sqlite3
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================= 公共工具函数 =================

def parse_float32_multiarray(blob: bytes):
    offset = 4
    dim_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    for _ in range(dim_len):
        label_len = struct.unpack_from("<I", blob, offset)[0]
        offset += 4 + label_len
        offset = (offset + 3) & ~3
        offset += 8
    offset += 4
    data_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    return struct.unpack_from(f"<{data_len}f", blob, offset)


def detrend_error(t, err):
    p = np.polyfit(t, err, 1)
    return err - (p[0] * t + p[1])


def compute_metrics(err):
    mse = np.mean(err**2)
    std = np.std(err)
    mx = np.max(np.abs(err))
    return mse, std, mx


# ================= 核心分析函数 =================

def analyze_one_scenario(bags, topic_name, warmup_sec, out_prefix):
    print(f"\n===== Analyzing {out_prefix.upper()} scenario =====")

    rows = []
    errors_for_box = {}

    for label, db_path in bags.items():
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
        row = cur.fetchone()
        if row is None:
            print(f"[WARN] {label}: topic not found, skipped")
            conn.close()
            continue

        topic_id = row[0]

        cur.execute("""
            SELECT timestamp, data FROM messages
            WHERE topic_id=? ORDER BY timestamp
        """, (topic_id,))

        t_list, err_list = [], []
        for ts, blob in cur.fetchall():
            t = ts * 1e-9
            values = parse_float32_multiarray(blob)
            err_ms = values[2] * 1e-3   # μs → ms
            t_list.append(t)
            err_list.append(err_ms)

        conn.close()

        t = np.array(t_list) - t_list[0]
        err = np.array(err_list)

        mask = t >= warmup_sec
        t = t[mask]
        err = err[mask]

        err_dt = detrend_error(t, err)
        errors_for_box[label] = err_dt

        mse, std, mx = compute_metrics(err_dt)
        rows.append({
            "Method": label,
            "MSE (ms^2)": mse,
            "STD (ms)": std,
            "MAX (ms)": mx
        })

        print(f"{label:10s} | MSE={mse:.3f}, STD={std:.3f}, MAX={mx:.3f}")

    # ===== 表格输出 =====
    df = pd.DataFrame(rows).round(3)
    df.to_csv(f"{out_prefix}_metrics.csv", index=False)
    df.to_latex(
        f"{out_prefix}_metrics.tex",
        index=False,
        float_format="%.3f",
        caption=f"{out_prefix.capitalize()} 场景下不同同步方法的定量性能对比",
        label=f"tab:{out_prefix}_metrics"
    )

    # ===== 箱线图 =====
    plt.figure(figsize=(5.5, 3.5))
    plt.boxplot(
        [errors_for_box[k] for k in errors_for_box.keys()],
        labels=list(errors_for_box.keys()),
        showfliers=False
    )
    plt.ylabel("Detrended Sync Error (ms)")
    plt.title(f"{out_prefix.capitalize()} Scenario")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_boxplot.pdf")
    plt.close()

    print(f"[OK] {out_prefix} tables & plots generated.")


# ================= 三个场景配置 =================

TOPIC_NAME = "/sync/status"
WARMUP_SEC = 10.0

# ---- Stable ----
BAGS_STABLE = {
    "FW": "/home/zhvv/aswrl_bags/stable_fw_001/stable_fw_001_0.db3",
    "Rule-ASW": "/home/zhvv/aswrl_bags/stable_rule_001/stable_rule_001_0.db3",
    "ASW-RL": "/home/zhvv/aswrl_bags/stable_aswrl_001/stable_aswrl_001_0.db3",
}

# ---- Jitter ----
BAGS_JITTER = {
    "FW": "/home/zhvv/aswrl_bags/jitter_fw_001/jitter_fw_001_0.db3",
    "Rule-ASW": "/home/zhvv/aswrl_bags/jitter_rule_001/jitter_rule_001_0.db3",
    "ASW-RL": "/home/zhvv/aswrl_bags/jitter_aswrl_001/jitter_aswrl_001_0.db3",
}

# ---- Multimodal（只有 FW / Rule-ASW）----
BAGS_MULTIMODAL = {
    "FW": "/home/zhvv/aswrl_bags/multimodal_fw_001/multimodal_fw_001_0.db3",
    "Rule-ASW": "/home/zhvv/aswrl_bags/multimodal_rule_001/multimodal_rule_001_0.db3",
}


# ================= 主程序 =================

if __name__ == "__main__":
    analyze_one_scenario(BAGS_STABLE, TOPIC_NAME, WARMUP_SEC, "stable")
    analyze_one_scenario(BAGS_JITTER, TOPIC_NAME, WARMUP_SEC, "jitter")
    analyze_one_scenario(BAGS_MULTIMODAL, TOPIC_NAME, WARMUP_SEC, "multimodal")

