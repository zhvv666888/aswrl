from rosbag2_sqlite_utils import iter_messages, find_topic_id
import struct

# 待检查的包
BAGS = {
    "Multimodal_RL": "/home/zhvv/aswrl_bags/multimodal_aswrl_001/multimodal_aswrl_001_0.db3",
    "Stable_Rule":   "/home/zhvv/aswrl_bags/stable_rule_001/stable_rule_001_0.db3"
}

TOPIC = "/formation/error"

def parse_float32_multi(blob):
    # 简易 Float32MultiArray 解析
    offset = 4 # CDR
    dim_len = struct.unpack_from("<I", blob, offset)[0]; offset += 4
    for _ in range(dim_len):
        label_len = struct.unpack_from("<I", blob, offset)[0]; offset += 4 + label_len
        offset = (offset + 3) & ~3; offset += 8
    offset += 4 # data_offset
    data_len = struct.unpack_from("<I", blob, offset)[0]; offset += 4
    if data_len == 0: return []
    return struct.unpack_from(f"<{data_len}f", blob, offset)

for label, path in BAGS.items():
    print(f"\nChecking {TOPIC} in {label}...")
    topic_id, _ = find_topic_id(path, TOPIC)
    
    if topic_id is None:
        print("  [ERROR] Topic not found!")
        continue
        
    count = 0
    for i, (t, blob) in enumerate(iter_messages(path, topic_id)):
        if count >= 3: break # 只看前3条
        try:
            vals = parse_float32_multi(blob)
            print(f"  Msg {i}: {vals}")
            count += 1
        except:
            print("  Parse error")
            
    if count == 0:
        print("  [WARN] Topic exists but NO MESSAGES found!")
