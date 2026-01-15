import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64MultiArray
from builtin_interfaces.msg import Time

from .clock_drift import SimClock
from .logging_utils import JsonlLogger, default_log_dir

class SyncMaster(Node):
    def __init__(self):
        super().__init__('sync_master')
        self.declare_parameter('sync_period', 0.1)
        self.declare_parameter('clients', ['/dog2', '/uav1'])
        self.declare_parameter('master_offset_ns', 0)
        self.declare_parameter('master_skew_ppm', 0.0)

        self.sync_period = float(self.get_parameter('sync_period').value)
        self.client_names = list(self.get_parameter('clients').value)

        self.clock_model = SimClock(
            offset_ns=int(self.get_parameter('master_offset_ns').value),
            skew_ppm=float(self.get_parameter('master_skew_ppm').value),
            noise_ns=0,
            seed=123
        )

        # publishers
        self.pub_req = self.create_publisher(Int64MultiArray, 'sync/request', 10)
        self.pub_follow = self.create_publisher(Int64MultiArray, 'sync/followup', 10)

        # subscribers
        self.sub_resp = self.create_subscription(Int64MultiArray, 'sync/response', self.on_response, 100)

        self.seq = 0
        self.pending = {}  # seq -> t1_local_ns
        self.timer = self.create_timer(self.sync_period, self.tick)

        log_dir = default_log_dir(self.get_name())
        self.logger_jsonl = JsonlLogger(f"{log_dir}/sync_master.jsonl")
        self.get_logger().info(f"Logging to {self.logger_jsonl.path}")

    def now_true_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def now_local_ns(self) -> int:
        return self.clock_model.local_time_ns(self.now_true_ns())

    def tick(self):
        self.seq += 1
        t1 = self.now_local_ns()
        self.pending[self.seq] = t1

        msg = Int64MultiArray()
        msg.data = [int(self.seq), int(t1)]
        self.pub_req.publish(msg)

        self.logger_jsonl.log({"event":"send_req", "seq": self.seq, "t1_ns": t1})

    def on_response(self, msg: Int64MultiArray):
        # response: [seq, t1, t2, t3]
        if len(msg.data) < 4:
            return
        seq, t1, t2, t3 = [int(x) for x in msg.data[:4]]
        if seq not in self.pending:
            return

        t4 = self.now_local_ns()
        follow = Int64MultiArray()
        follow.data = [seq, int(t4)]
        self.pub_follow.publish(follow)

        self.logger_jsonl.log({
            "event":"recv_resp_send_followup",
            "seq": seq,
            "t1_ns": int(t1), "t2_ns": int(t2), "t3_ns": int(t3), "t4_ns": int(t4)
        })

    def destroy_node(self):
        try:
            self.logger_jsonl.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = SyncMaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

