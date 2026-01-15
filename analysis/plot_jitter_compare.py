import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt


# ====== Jitter 场景数据路径 ======
BAGS = {
    "FW": "/home/zhvv/aswrl_bags/jitter_fw_001/jitter_fw_001_0.db3",
    "Rule-ASW": "/home/zhvv/aswrl_bags/jitter_rule_001/jitter_rule_001_0.db3",
    "ASW-RL": "/home/zhvv/aswrl_bags/jitter_aswrl_001/jitter_aswrl_001_0.db3",
}

TOPIC_NAME = "/sync/status"
WARMUP_SEC = 10.0


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


def load_sync_error_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id FROM topics WHERE name=?", (TOPIC_NAME,))
    topic_id = cur.fetchone()[0]
    cur.execute("""
        SELECT timestamp, data FROM messages
        WHERE topic_id=? ORDER BY timestamp
    """, (topic_id,))
    t_list, err_list = [], []
    for ts, blob in cur.fetchall():
        t = ts * 1e-9
        values = parse_float32_multiarray(blob)
        err_ms = values[2] * 1e-3
        t_list.append(t)
        err_list.append(err_ms)
    conn.close()
    t = np.array(t_list) - t_list[0]
    err = np.array(err_list)
    mask = t >= WARMUP_SEC
    return t[mask], err[mask]


def detrend_error(t, err):
    p = np.polyfit(t, err, 1)
    return err - (p[0] * t + p[1])


def compute_metrics(err):
    return np.mean(err**2), np.std(err), np.max(np.abs(err))


# ====== 1. 去趋势误差时间序列 ======
plt.figure(figsize=(6.8, 3.2))
errors_for_box = {}

for label, db in BAGS.items():
    t, err = load_sync_error_from_db(db)
    err_dt = detrend_error(t, err)
    errors_for_box[label] = err_dt
    plt.plot(t, err_dt, label=label, linewidth=1.4, alpha=0.8)

plt.xlabel("Time (s)")
plt.ylabel("Detrended Sync Error (ms)")
plt.title("Jitter Network Scenario")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ====== 2. 箱线图 + 指标 ======
plt.figure(figsize=(5.5, 3.5))
plt.boxplot(
    [errors_for_box[k] for k in ["FW", "Rule-ASW", "ASW-RL"]],
    labels=["FW", "Rule-ASW", "ASW-RL"],
    showfliers=False,
)
plt.ylabel("Detrended Sync Error (ms)")
plt.title("Quantitative Comparison under Jitter Scenario")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("=== Jitter Scenario Metrics ===")
for k, v in errors_for_box.items():
    mse, std, mx = compute_metrics(v)
    print(f"{k:10s} | MSE={mse:.3f}, STD={std:.3f}, MAX={mx:.3f}")
