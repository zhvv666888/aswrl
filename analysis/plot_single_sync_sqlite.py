import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt


BAG_DB = "/home/zhvv/aswrl_bags/stable_fw_001/stable_fw_001_0.db3"
TOPIC_NAME = "/sync/status"

WARMUP_SEC = 10.0
UNIT_SCALE = 1e3   # s -> ms


def parse_float32_multiarray(blob: bytes):
    """
    Parse std_msgs/msg/Float32MultiArray from ROS2 CDR buffer
    """
    # ⚠️ 关键：跳过 ROS2 CDR encapsulation header
    offset = 4

    # ---------- layout.dim ----------
    # uint32 dim_length
    dim_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4

    # each MultiArrayDimension
    for _ in range(dim_len):
        # string label
        label_len = struct.unpack_from("<I", blob, offset)[0]
        offset += 4
        offset += label_len

        # padding to 4 bytes
        offset = (offset + 3) & ~3

        # uint32 size + uint32 stride
        offset += 8

    # ---------- data_offset ----------
    offset += 4  # uint32

    # ---------- data ----------
    data_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4

    values = struct.unpack_from(f"<{data_len}f", blob, offset)
    return values


def load_sync_error_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 找 topic_id
    cur.execute("SELECT id FROM topics WHERE name=?", (TOPIC_NAME,))
    row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"Topic {TOPIC_NAME} not found")
    topic_id = row[0]

    # 读取消息
    cur.execute("""
        SELECT timestamp, data
        FROM messages
        WHERE topic_id=?
        ORDER BY timestamp
    """, (topic_id,))

    t_list = []
    err_list = []

    rows = cur.fetchall()
    for i, (ts, blob) in enumerate(rows):
        t = ts * 1e-9
        values = parse_float32_multiarray(blob)
        sync_err = values[2] * 1e-3   # 微秒 → 毫秒

        t_list.append(t)
        err_list.append(sync_err)

    conn.close()

    t = np.array(t_list)
    err = np.array(err_list)

    # 时间从 0 开始
    t = t - t[0]

    # 去掉 warm-up
    mask = t >= WARMUP_SEC
    return t[mask], err[mask]


if __name__ == "__main__":
    t, err = load_sync_error_from_db(BAG_DB)

    plt.figure(figsize=(6, 3))
    plt.plot(t, err * UNIT_SCALE, linewidth=1.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Sync Error (ms)")
    plt.title("Stable + Baseline-FW")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

