import sqlite3
import struct
import numpy as np
import matplotlib.pyplot as plt


BAG_DB = "/home/zhvv/aswrl_bags/stable_fw_001/stable_fw_001_0.db3"
TOPIC_NAME = "/sync/status"

WARMUP_SEC = 10.0
UNIT_SCALE = 1e3   # s -> ms


def load_sync_error_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. 找 topic_id
    cur.execute("SELECT id FROM topics WHERE name=?", (TOPIC_NAME,))
    topic_id = cur.fetchone()[0]

    # 2. 读 timestamp + data
    cur.execute("""
        SELECT timestamp, data
        FROM messages
        WHERE topic_id=?
        ORDER BY timestamp
    """, (topic_id,))

    t_list = []
    err_list = []

    for ts, blob in cur.fetchall():
        # rosbag timestamp: ns
        t = ts * 1e-9

        # std_msgs/Float32MultiArray
        # CDR layout:
        # uint32 length + float32 * length
        length = struct.unpack_from("<I", blob, 0)[0]
        values = struct.unpack_from(f"<{length}f", blob, 4)

        sync_err = values[0]   # 第 0 个：同步偏差（秒）

        t_list.append(t)
        err_list.append(sync_err)

    conn.close()

    t = np.array(t_list)
    err = np.array(err_list)

    # 时间从 0 开始
    t = t - t[0]

    # 去 warm-up
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

