"""
Figure 5.7 - RTT distribution / switching illustration (paper-ready)
====================================================================

Outputs:
1) fig_5_7_delay_histograms.png  : histograms (Stable / Jitter / Multimodal)
2) fig_5_7_delay_timeseries.png  : time series (Stable / Jitter / Multimodal)

Preferred: use a network-status topic (e.g., /net_state).
Fallback: derive RTT proxy from /sync/request and /sync/response timestamps.

This "paper-ready" version:
- uses SEQ pairing by default (avoids 0ms artifacts due to high request rate),
- fixes histogram x-limits and bins for fair comparison,
- down-samples time-series for readability,
- labels as RTT (ms),
- annotates <1ms proportion per subplot.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from rosbag2_sqlite_utils import (
    best_topic_by_keywords,
    list_topics,
    find_topic_id,
    iter_messages,
    load_multiarray_index_series,
    load_scalar_series,
    parse_std_float32,
)

# -----------------------------
# Configuration
# -----------------------------
WARMUP_SEC = 0.0

USE_CSV = False
CSV_PATHS = {
    "Stable": "stable_net_delay.csv",
    "Jitter": "jitter_net_delay.csv",
    "Multimodal": "multimodal_net_delay.csv",
}

BAG_PATHS = {
    "Stable": "/home/zhvv/aswrl_bags/stable_fw_001/stable_fw_001_0.db3",
    "Jitter": "/home/zhvv/aswrl_bags/jitter_fw_001/jitter_fw_001_0.db3",
    "Multimodal": "/home/zhvv/aswrl_bags/multimodal_fw_001/multimodal_fw_001_0.db3",
}

NET_TOPIC_CANDIDATES = ["/net_state", "/network_state", "/net/state", "/netstat", "/net/status"]
DELAY_INDEX = 0
DELAY_UNIT_SCALE = 1.0  # assumes ms already

# Fallback (/sync/request <-> /sync/response)
FALLBACK_METHOD = "SEQ"  # "SEQ" recommended
SEQ_INDEX: Optional[int] = 0  # FIXED: use 0 based on your logs
MAX_SAMPLES_FOR_GUESS = 300

# Plot settings (paper-ready)
HIST_XLIM = (0.0, 150.0)                      # show up to 150ms (covers multimodal max ~144ms)
HIST_BINS = np.linspace(HIST_XLIM[0], HIST_XLIM[1], 76)  # 75 bins
TS_DOWNSAMPLE_STEP = 5                        # readability
ANNOT_THRESH_MS = 1.0                         # show "<1ms" proportion


# -----------------------------
# Loaders
# -----------------------------
def load_delay_series_from_csv(csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("time_sec") or cols.get("t") or cols.get("time") or df.columns[0]
    dcol = cols.get("delay_ms") or cols.get("delay") or df.columns[1]
    t = df[tcol].to_numpy(dtype=float)
    d = df[dcol].to_numpy(dtype=float)

    t = t - float(t[0])
    m = t >= WARMUP_SEC
    return t[m], d[m]


def pick_net_topic(db_path: str) -> str:
    topics = [t.name for t in list_topics(db_path)]
    for cand in NET_TOPIC_CANDIDATES:
        if cand in topics:
            return cand

    best = best_topic_by_keywords(db_path, keywords=["net", "state", "delay", "jitter", "loss"])
    if best is None:
        raise RuntimeError(
            f"Cannot find a network-status topic in {db_path}. "
            f"Topics found: {topics[:20]} ... (total {len(topics)})"
        )
    return best[1]


def print_topics(db_path: str, max_n: int = 60):
    print(f"\n[TOPICS] {db_path}")
    topics = list_topics(db_path)
    for t in topics[:max_n]:
        print(f"  - {t.name} ({t.type})")
    if len(topics) > max_n:
        print(f"  ... ({len(topics) - max_n} more)")


# ---- Int64MultiArray parsing + seq-pair fallback ----
def parse_int64_multiarray(blob: bytes):
    """
    Parse std_msgs/msg/Int64MultiArray from ROS2 CDR buffer.
    """
    offset = 4  # CDR encapsulation
    dim_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4

    for _ in range(dim_len):
        label_len = struct.unpack_from("<I", blob, offset)[0]
        offset += 4 + label_len
        offset = (offset + 3) & ~3  # pad to 4
        offset += 8  # uint32 size + uint32 stride

    offset += 4  # data_offset
    data_len = struct.unpack_from("<I", blob, offset)[0]
    offset += 4
    offset = (offset + 7) & ~7  # align to 8 for int64
    return struct.unpack_from(f"<{data_len}q", blob, offset)  # signed int64


def _guess_seq_index(req_datas, resp_datas, topk: int = 3) -> Optional[int]:
    if not req_datas:
        return None

    min_len = min(len(x) for x in req_datas)
    if resp_datas:
        min_len = min(min_len, min(len(x) for x in resp_datas))

    scores = []
    for idx in range(min_len):
        seq = [x[idx] for x in req_datas]
        diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
        if not diffs:
            continue

        inc01 = sum(1 for d in diffs if d in (0, 1))
        nondec = sum(1 for d in diffs if d >= 0)
        uniq = len(set(seq)) / max(1, len(seq))

        score = 0.55 * (inc01 / len(diffs)) + 0.35 * (nondec / len(diffs)) + 0.10 * uniq
        scores.append((score, idx))

    if not scores:
        return None
    scores.sort(reverse=True)
    print("[SEQ-GUESS] Top candidates (score, index):", scores[:topk])
    return scores[0][1]


def load_delay_series_from_sync_seq_pair(
    db_path: str,
    request_topic: str = "/sync/request",
    response_topic: str = "/sync/response",
    seq_index: Optional[int] = None,
    max_samples_for_guess: int = 300,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    req_id, req_name = find_topic_id(db_path, request_topic, fuzzy=True)
    resp_id, resp_name = find_topic_id(db_path, response_topic, fuzzy=True)

    # guess seq_index if needed
    if seq_index is None:
        req_samples = []
        resp_samples = []
        for k, (_t, blob) in enumerate(iter_messages(db_path, req_id)):
            req_samples.append(parse_int64_multiarray(blob))
            if k + 1 >= max_samples_for_guess:
                break
        for k, (_t, blob) in enumerate(iter_messages(db_path, resp_id)):
            resp_samples.append(parse_int64_multiarray(blob))
            if k + 1 >= max_samples_for_guess:
                break
        seq_index = _guess_seq_index(req_samples, resp_samples)

    if seq_index is None:
        raise RuntimeError("Cannot guess seq_index from Int64MultiArray.data. Please set SEQ_INDEX manually.")

    print(f"[INFO] Fallback SEQ-pair: seq_index={seq_index} for {req_name}/{resp_name}")

    # build request map
    req_time = {}
    for t, blob in iter_messages(db_path, req_id):
        data = parse_int64_multiarray(blob)
        if seq_index >= len(data):
            continue
        seq = int(data[seq_index])
        if seq not in req_time:
            req_time[seq] = t

    # match responses
    t0 = None
    t_rel, delays_ms = [], []
    for t, blob in iter_messages(db_path, resp_id):
        data = parse_int64_multiarray(blob)
        if seq_index >= len(data):
            continue
        seq = int(data[seq_index])
        if seq not in req_time:
            continue

        dt = t - req_time.pop(seq)  # seconds
        if dt < 0:
            continue

        if t0 is None:
            t0 = t
        trel = t - t0
        if trel < WARMUP_SEC:
            continue

        t_rel.append(trel)
        delays_ms.append(dt * 1000.0)

    if not t_rel:
        raise RuntimeError("No matched request/response pairs found by seq. Check SEQ_INDEX or message layout.")
    return np.asarray(t_rel), np.asarray(delays_ms), f"{req_name}<->{resp_name}", "SEQ-PAIR"


def load_delay_series_from_sync_rtt(
    db_path: str,
    request_topic: str = "/sync/request",
    response_topic: str = "/sync/response",
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    req_id, req_name = find_topic_id(db_path, request_topic, fuzzy=True)
    resp_id, resp_name = find_topic_id(db_path, response_topic, fuzzy=True)

    req_ts = [t for t, _ in iter_messages(db_path, req_id)]
    resp_ts = [t for t, _ in iter_messages(db_path, resp_id)]
    if not req_ts or not resp_ts:
        raise RuntimeError(f"RTT fallback failed: empty {req_name} or {resp_name} in {db_path}")

    req_ts = np.asarray(req_ts, dtype=float)
    resp_ts = np.asarray(resp_ts, dtype=float)

    t0 = resp_ts[0]
    t_rel, delays_ms = [], []
    i = 0
    for tr in resp_ts:
        while i + 1 < len(req_ts) and req_ts[i + 1] <= tr:
            i += 1
        if req_ts[i] > tr:
            continue
        dt = tr - req_ts[i]
        t = tr - t0
        if t < WARMUP_SEC:
            continue
        t_rel.append(t)
        delays_ms.append(dt * 1000.0)

    return np.asarray(t_rel), np.asarray(delays_ms), f"{req_name}->{resp_name}", "RTT"


def load_delay_series_from_bag(db_path: str):
    try:
        topic = pick_net_topic(db_path)
        print(f"[INFO] Using network topic: {topic}")
    except RuntimeError as e:
        print(f"[WARN] {e}")
        if FALLBACK_METHOD.upper() == "SEQ":
            print("[WARN] Falling back to SEQ-paired delay from /sync/request <-> /sync/response.")
            return load_delay_series_from_sync_seq_pair(
                db_path,
                seq_index=SEQ_INDEX,
                max_samples_for_guess=MAX_SAMPLES_FOR_GUESS,
            )
        else:
            print("[WARN] Falling back to naive RTT delay from /sync/request -> /sync/response.")
            return load_delay_series_from_sync_rtt(db_path)

    # parse scalar or multiarray
    try:
        t, y, real = load_scalar_series(
            db_path=db_path,
            topic=topic,
            parser=parse_std_float32,
            warmup_sec=WARMUP_SEC,
            unit_scale=DELAY_UNIT_SCALE,
            fuzzy_topic=False,
        )
        return t, y, real, "Float32"
    except Exception as e_float:
        try:
            t, y, real = load_multiarray_index_series(
                db_path=db_path,
                topic=topic,
                index=DELAY_INDEX,
                warmup_sec=WARMUP_SEC,
                unit_scale=DELAY_UNIT_SCALE,
                fuzzy_topic=False,
            )
            return t, y, real, f"Float32MultiArray[{DELAY_INDEX}]"
        except Exception as e_arr:
            print("[ERROR] Failed to parse as Float32 and Float32MultiArray.")
            print("Float32 error:", repr(e_float))
            print("MultiArray error:", repr(e_arr))
            raise


# -----------------------------
# Plot helpers (paper-ready)
# -----------------------------
def summarize_delay(name: str, d: np.ndarray):
    d = np.asarray(d)
    p0 = np.mean(d < ANNOT_THRESH_MS) * 100
    print(
        f"[{name}] n={len(d)}  min={d.min():.3f}  p1={np.percentile(d,1):.3f}  "
        f"median={np.median(d):.3f}  mean={d.mean():.3f}  p99={np.percentile(d,99):.3f}  "
        f"max={d.max():.3f}  (<{ANNOT_THRESH_MS:.0f}ms)={p0:.2f}%"
    )
    return p0


def annotate_p0(ax, p0: float):
    ax.text(
        0.98,
        0.95,
        f"<{ANNOT_THRESH_MS:.0f}ms: {p0:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, linewidth=0.8),
    )


def plot_hist(ax, delay_ms: np.ndarray, title: str, p0: float):
    ax.hist(delay_ms, bins=HIST_BINS, density=True)
    ax.set_title(title)
    ax.set_xlabel("RTT (ms)")
    ax.set_ylabel("Density")
    ax.set_xlim(*HIST_XLIM)
    ax.grid(True, alpha=0.3)
    annotate_p0(ax, p0)


def plot_timeseries(ax, t: np.ndarray, delay_ms: np.ndarray, title: str, p0: float):
    step = max(1, int(TS_DOWNSAMPLE_STEP))
    ax.plot(t[::step], delay_ms[::step], linewidth=1.0, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RTT (ms)")
    ax.grid(True, alpha=0.3)
    annotate_p0(ax, p0)


# -----------------------------
# Main
# -----------------------------
def main():
    out_dir = Path(".")
    delays = {}      # scenario -> (t, d, meta_topic, kind)
    p0s = {}         # scenario -> <1ms proportion

    for scenario in ["Stable", "Jitter", "Multimodal"]:
        if USE_CSV:
            t, d = load_delay_series_from_csv(CSV_PATHS[scenario])
            p0 = summarize_delay(scenario, d)
            delays[scenario] = (t, d, "csv", "csv")
            p0s[scenario] = p0
        else:
            db = BAG_PATHS[scenario]
            print_topics(db)
            t, d, real_topic, kind = load_delay_series_from_bag(db)
            p0 = summarize_delay(scenario, d)
            delays[scenario] = (t, d, real_topic, kind)
            p0s[scenario] = p0

    # 1) histograms (paper-ready)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    for ax, scenario in zip(axes, ["Stable", "Jitter", "Multimodal"]):
        _, d, _, _ = delays[scenario]
        plot_hist(ax, d, scenario, p0s[scenario])
    fig.tight_layout()
    fig.savefig(out_dir / "fig_5_7_delay_histograms.png", dpi=300)
    print("[OK] Saved fig_5_7_delay_histograms.png")

    # 2) time series (paper-ready)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=False)
    for ax, scenario in zip(axes, ["Stable", "Jitter", "Multimodal"]):
        t, d, _, _ = delays[scenario]
        plot_timeseries(ax, t, d, scenario, p0s[scenario])
    fig.tight_layout()
    fig.savefig(out_dir / "fig_5_7_delay_timeseries.png", dpi=300)
    print("[OK] Saved fig_5_7_delay_timeseries.png")


if __name__ == "__main__":
    main()

