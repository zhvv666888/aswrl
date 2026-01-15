"""
rosbag2_sqlite_utils.py
======================
Utility functions to read ROS2 rosbag2 SQLite (.db3) files and parse common std_msgs
types without needing ROS2 runtime.

This module is written to be consistent with your existing scripts:
- sync status topic: /sync/status (std_msgs/msg/Float32MultiArray)
- sync error is typically stored at values[2] in microseconds, then converted to ms
  (as you already do in plot_stable_compare.py / plot_jitter_compare.py / plot_multimodal_compare.py).

If your message layout differs (e.g., delay index / window index), change the INDEX
constants in the figure scripts (or rely on the auto-guess helpers).
"""
from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------
# Basic DB helpers
# ----------------------------
@dataclass
class TopicInfo:
    id: int
    name: str
    type: Optional[str] = None


def list_topics(db_path: str) -> List[TopicInfo]:
    """
    Return topics defined in rosbag2 sqlite file.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # rosbag2 topics schema usually has: id, name, type, serialization_format, offered_qos_profiles
    # but we keep it robust if 'type' is missing.
    try:
        cur.execute("SELECT id, name, type FROM topics ORDER BY id")
        rows = cur.fetchall()
        topics = [TopicInfo(int(r[0]), str(r[1]), str(r[2])) for r in rows]
    except sqlite3.OperationalError:
        cur.execute("SELECT id, name FROM topics ORDER BY id")
        rows = cur.fetchall()
        topics = [TopicInfo(int(r[0]), str(r[1]), None) for r in rows]

    conn.close()
    return topics


def find_topic_id(db_path: str, topic_name: str, fuzzy: bool = False) -> Tuple[int, str]:
    """
    Find topic_id by exact name, or by fuzzy LIKE match when fuzzy=True.
    Returns (topic_id, resolved_topic_name).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not fuzzy:
        cur.execute("SELECT id, name FROM topics WHERE name=?", (topic_name,))
        row = cur.fetchone()
        if row is None:
            conn.close()
            raise RuntimeError(f"Topic not found: {topic_name} in {db_path}")
        conn.close()
        return int(row[0]), str(row[1])

    # fuzzy match
    pat = f"%{topic_name.strip('%')}%"
    cur.execute("SELECT id, name FROM topics WHERE name LIKE ? LIMIT 1", (pat,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise RuntimeError(f"No topic LIKE '{pat}' found in {db_path}")
    conn.close()
    return int(row[0]), str(row[1])


def best_topic_by_keywords(db_path: str, keywords: Sequence[str]) -> Optional[Tuple[int, str]]:
    """
    Pick the topic with the highest keyword hit count in its name.
    """
    topics = list_topics(db_path)
    best = None
    best_score = 0
    for t in topics:
        name_l = t.name.lower()
        score = sum(1 for k in keywords if k.lower() in name_l)
        if score > best_score:
            best_score = score
            best = (t.id, t.name)
    return best


def iter_messages(db_path: str, topic_id: int) -> Iterable[Tuple[float, bytes]]:
    """
    Yield (t_sec, data_blob) ordered by timestamp for a given topic_id.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT timestamp, data
        FROM messages
        WHERE topic_id=?
        ORDER BY timestamp
        """,
        (topic_id,),
    )
    for ts, blob in cur.fetchall():
        yield float(ts) * 1e-9, blob
    conn.close()


# ----------------------------
# CDR parsers for std_msgs
# ----------------------------
def parse_float32_multiarray(blob: bytes) -> Tuple[float, ...]:
    """
    Parse std_msgs/msg/Float32MultiArray from ROS2 CDR buffer.

    NOTE: This implementation matches your stable/jitter/multimodal scripts:
    - Skip 4-byte CDR encapsulation header
    - Parse layout.dim (string label + padding + size/stride)
    - Parse data_offset
    - Parse data length and float array
    """
    offset = 4  # CDR encapsulation
    dim_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4

    for _ in range(dim_len):
        label_len = struct.unpack_from("<I", blob, offset)[0]
        offset += 4 + label_len
        offset = (offset + 3) & ~3  # padding to 4 bytes
        offset += 8  # uint32 size + uint32 stride

    offset += 4  # data_offset
    data_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    return struct.unpack_from(f"<{data_len}f", blob, offset)


def parse_std_float32(blob: bytes) -> float:
    """
    Parse std_msgs/msg/Float32 from ROS2 CDR buffer.
    """
    offset = 4  # CDR encapsulation
    return struct.unpack_from("<f", blob, offset)[0]


def parse_std_int32(blob: bytes) -> int:
    """
    Parse std_msgs/msg/Int32 from ROS2 CDR buffer.
    """
    offset = 4  # CDR encapsulation
    return int(struct.unpack_from("<i", blob, offset)[0])


# ----------------------------
# High-level loaders
# ----------------------------
def load_multiarray_index_series(
    db_path: str,
    topic: str,
    index: int,
    warmup_sec: float = 0.0,
    unit_scale: float = 1.0,
    fuzzy_topic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load a time series from Float32MultiArray[index].
    Returns (t, y, resolved_topic_name).
    """
    topic_id, real_name = find_topic_id(db_path, topic, fuzzy=fuzzy_topic)
    t_list, y_list = [], []
    t0 = None

    for t_sec, blob in iter_messages(db_path, topic_id):
        if t0 is None:
            t0 = t_sec
        t = t_sec - t0
        if t < warmup_sec:
            continue
        values = parse_float32_multiarray(blob)
        if index >= len(values):
            raise RuntimeError(f"Index {index} out of range (len={len(values)}) for topic {real_name}")
        t_list.append(t)
        y_list.append(values[index] * unit_scale)

    return np.asarray(t_list), np.asarray(y_list), real_name


def load_scalar_series(
    db_path: str,
    topic: str,
    parser: Callable[[bytes], float],
    warmup_sec: float = 0.0,
    unit_scale: float = 1.0,
    fuzzy_topic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load a scalar series from a topic whose message is a single number (Float32/Int32).
    Returns (t, y, resolved_topic_name).
    """
    topic_id, real_name = find_topic_id(db_path, topic, fuzzy=fuzzy_topic)
    t_list, y_list = [], []
    t0 = None

    for t_sec, blob in iter_messages(db_path, topic_id):
        if t0 is None:
            t0 = t_sec
        t = t_sec - t0
        if t < warmup_sec:
            continue
        y = parser(blob) * unit_scale
        t_list.append(t)
        y_list.append(y)

    return np.asarray(t_list), np.asarray(y_list), real_name


def detrend_linear(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Remove linear trend via y - (a t + b).
    Matches your existing detrend implementation. (polyfit degree=1)
    """
    if len(t) < 2:
        return y.copy()
    p = np.polyfit(t, y, 1)
    return y - (p[0] * t + p[1])


def compute_metrics(err_ms: np.ndarray) -> Tuple[float, float, float]:
    """
    Metrics used in your tables:
    - MSE (ms^2)
    - STD (ms)
    - MAX (ms) of absolute error
    """
    mse = float(np.mean(err_ms ** 2))
    std = float(np.std(err_ms))
    mx = float(np.max(np.abs(err_ms)))
    return mse, std, mx


def guess_window_index_from_status(
    db_path: str,
    status_topic: str,
    warmup_sec: float = 0.0,
    max_samples: int = 1500,
) -> List[Tuple[int, float, dict]]:
    """
    Heuristically guess which Float32MultiArray index corresponds to window length.

    Returns a ranked list: [(index, score, info_dict), ...] highest score first.
    You should still verify by printing the first few values.

    Heuristic:
    - window length should be positive and not crazy large
    - often close to integers
    - relatively smooth compared to raw error
    """
    topic_id, real_name = find_topic_id(db_path, status_topic, fuzzy=False)
    t0 = None
    rows = []
    for i, (t_sec, blob) in enumerate(iter_messages(db_path, topic_id)):
        if t0 is None:
            t0 = t_sec
        t = t_sec - t0
        if t < warmup_sec:
            continue
        vals = parse_float32_multiarray(blob)
        rows.append(vals)
        if len(rows) >= max_samples:
            break

    if not rows:
        raise RuntimeError(f"No samples found after warmup_sec={warmup_sec} in {db_path}")

    arr = np.array(rows, dtype=float)  # shape: (N, D)
    D = arr.shape[1]
    ranked: List[Tuple[int, float, dict]] = []
    for j in range(D):
        x = arr[:, j]
        x = x[np.isfinite(x)]
        if len(x) < 10:
            continue
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmax <= 0:
            continue

        # integer-likeness
        frac = np.abs(x - np.round(x))
        int_like = float(np.mean(frac < 1e-3))

        # smoothness: smaller median absolute diff is smoother
        mad_diff = float(np.median(np.abs(np.diff(x))))

        # range penalty
        range_penalty = 0.0
        if xmax > 20000:
            range_penalty += 1.5
        elif xmax > 5000:
            range_penalty += 0.8
        if xmin < 0:
            range_penalty += 1.0

        score = (
            2.0 * int_like
            - 0.02 * mad_diff
            - range_penalty
            - 0.00005 * xmax
        )

        info = {
            "topic": real_name,
            "min": xmin,
            "max": xmax,
            "int_like_ratio": int_like,
            "median_abs_diff": mad_diff,
        }
        ranked.append((j, score, info))

    ranked.sort(key=lambda z: z[1], reverse=True)
    return ranked
