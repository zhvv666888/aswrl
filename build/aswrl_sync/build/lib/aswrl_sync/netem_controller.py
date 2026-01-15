import os
import time
import subprocess
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .logging_utils import JsonlLogger, default_log_dir

class NetemController(Node):
    """Network disturbance emulator.

    Reads a YAML scenario with time segments and applies `tc qdisc replace ... netem ...`.
    Must be run with sufficient privileges (usually via `sudo -E`), and *inside* the target netns.

    Params:
      - iface: e.g., veth_ns_dog2_n
      - scenario_yaml: path to scenario yaml
    Publishes:
      - net_state: [delay_ms, jitter_ms, loss_pct, segment_id]
    """
    def __init__(self):
        super().__init__('netem_controller')
        self.declare_parameter('iface', 'veth_ns_dog2_n')
        self.declare_parameter('scenario_yaml', '')
        self.declare_parameter('loop', True)

        self.iface = str(self.get_parameter('iface').value)
        self.scenario_yaml = str(self.get_parameter('scenario_yaml').value)
        self.loop = bool(self.get_parameter('loop').value)

        if not self.scenario_yaml or not os.path.exists(self.scenario_yaml):
            raise RuntimeError(f"scenario_yaml not found: {self.scenario_yaml}")

        with open(self.scenario_yaml, 'r', encoding='utf-8') as f:
            self.sc = yaml.safe_load(f)

        self.period_s = float(self.sc.get('period_s', 60))
        self.segments = list(self.sc.get('segments', []))
        if not self.segments:
            raise RuntimeError("No segments found in scenario yaml.")

        self.t0 = time.time()

        self.pub = self.create_publisher(Float32MultiArray, 'net_state', 10)
        self.timer = self.create_timer(0.2, self.tick)

        log_dir = default_log_dir(self.get_name())
        self.logger_jsonl = JsonlLogger(f"{log_dir}/netem_controller.jsonl")
        self.get_logger().info(f"Logging to {self.logger_jsonl.path}")

    def apply_netem(self, delay_ms: float, jitter_ms: float, loss_pct: float):
        # Use `tc qdisc replace` so reruns are idempotent.
        cmd = [
            "tc", "qdisc", "replace", "dev", self.iface, "root", "netem",
            "delay", f"{delay_ms:.3f}ms", f"{jitter_ms:.3f}ms", "distribution", "normal",
            "loss", f"{loss_pct:.3f}%"
        ]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def current_segment(self, t: float):
        # t in [0, period)
        seg_id = 0
        cur = self.segments[0]
        for i, seg in enumerate(self.segments):
            if float(seg.get('t0', 0)) <= t:
                seg_id = i
                cur = seg
        return seg_id, cur

    def tick(self):
        elapsed = time.time() - self.t0
        if self.loop:
            t = elapsed % self.period_s
        else:
            t = min(elapsed, self.period_s - 1e-6)

        seg_id, seg = self.current_segment(t)
        delay_ms = float(seg.get('delay_ms', 20))
        jitter_ms = float(seg.get('jitter_ms', 0))
        loss_pct = float(seg.get('loss_pct', 0))

        self.apply_netem(delay_ms, jitter_ms, loss_pct)

        msg = Float32MultiArray()
        msg.data = [float(delay_ms), float(jitter_ms), float(loss_pct), float(seg_id)]
        self.pub.publish(msg)

        self.logger_jsonl.log({"event":"netem", "seg": seg_id, "delay_ms": delay_ms, "jitter_ms": jitter_ms, "loss_pct": loss_pct})

def main():
    rclpy.init()
    node = NetemController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
