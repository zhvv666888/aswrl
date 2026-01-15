import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

def default_log_dir(node_name: str) -> str:
    ts = time.strftime("%Y%m%d")
    base = os.path.expanduser(f"~/.aswrl_logs/{node_name}/{ts}")
    os.makedirs(base, exist_ok=True)
    return base

class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(self.path, "a", encoding="utf-8")

    def log(self, record: Dict[str, Any]):
        record = dict(record)
        record.setdefault("ts_unix", time.time())
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

def to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": obj}
