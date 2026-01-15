import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt


# ====== 数据路径 ======
BAGS = {
    "FW": "/home/zhvv/aswrl_bags/stable_fw_001/stable_fw_001_0.db3",
    "Rule-ASW": "/home/zhvv/aswrl_bags/stable_rule_001/stable_rule_001_0.db3",
    "ASW-RL": "/home/zhvv/aswrl_bags/stable_aswrl_001/stable_aswrl_001_0.db3",
}

TOPIC_NAME = "/sync/status"
WARMUP_SEC = 10.0   # 前 10 秒不画


# ====== ROS2 CDR 解析 ======
def parse_float32_multiarray(blob: bytes):
    offset = 4  # 跳过 CDR encapsulation header

    dim_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4

    for _ in range(dim_len):
        label_len = struct.unpack_from("<I", blob, offset)[0]
        offset += 4 + label_len
        offset = (offset + 3) & ~3  # padding
        offset += 8  # size + stride

    offset += 4  # data_offset

    data_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4

    values = struct.unpack_from(f"<{data_len}f", blob, offset)
    return values


# ====== 读取一个 bag ======
def load_sync_error_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id FROM topics WHERE name=?", (TOPIC_NAME,))
    topic_id = cur.fetchone()[0]

    cur.execute("""
        SELECT timestamp, data
        FROM messages
        WHERE topic_id=?
        ORDER BY timestamp
    """, (topic_id,))

    t_list = []
    err_list = []

    for ts, blob in cur.fetchall():
        t = ts * 1e-9
        values = parse_float32_multiarray(blob)

        # ⭐ 同步误差在 index = 2，单位：微秒
        sync_err_ms = values[2] * 1e-3

        t_list.append(t)
        err_list.append(sync_err_ms)

    conn.close()

    t = np.array(t_list)
    err = np.array(err_list)

    t = t - t[0]
    mask = t >= WARMUP_SEC
    return t[mask], err[mask]


# ====== 去线性趋势（方案 A 核心）=====
def detrend_error(t, err):
    """
    去除一阶线性趋势：e(t) - (a t + b)
    """
    p = np.polyfit(t, err, 1)
    trend = p[0] * t + p[1]
    return err - trend
    
def compute_metrics(err):
    """
    err: 去趋势后的同步误差（ms）
    """
    mse = np.mean(err ** 2)
    std = np.std(err)
    max_err = np.max(np.abs(err))
    return mse, std, max_err
# ====== 主程序：画三策略同图 ======
metrics = {}
errors_for_boxplot = {}

for label, db in BAGS.items():
    t, err = load_sync_error_from_db(db)
    err_dt = detrend_error(t, err)

    mse, std, max_err = compute_metrics(err_dt)
    metrics[label] = (mse, std, max_err)

    errors_for_boxplot[label] = err_dt

# 打印数值（可以直接贴进论文表格草稿）
print("=== Quantitative Metrics (Detrended Error, ms) ===")
for label, (mse, std, max_err) in metrics.items():
    print(f"{label:10s} | MSE = {mse:.3f}, STD = {std:.3f}, MAX = {max_err:.3f}")


plt.figure(figsize=(5.5, 3.5))

data = [
    errors_for_boxplot["FW"],
    errors_for_boxplot["Rule-ASW"],
    errors_for_boxplot["ASW-RL"],
]

plt.boxplot(
    data,
    labels=["FW", "Rule-ASW", "ASW-RL"],
    showfliers=False,   # 不画极端离群点，论文里常见
)
plt.yscale("symlog", linthresh=1)
plt.ylabel("Detrended Sync Error (ms)")
plt.title("Quantitative Comparison under Stable Scenario")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


