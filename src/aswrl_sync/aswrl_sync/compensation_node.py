import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from .logging_utils import JsonlLogger, default_log_dir

class CompensationNode(Node):
    """Compensate timestamps for incoming odometry based on sync status.

    Subscribe:
      - odom_local (from local robot)
      - sync/status: [offset_est_ns, delay_est_ns, drift, window, sync_err]
    Publish:
      - odom_synced: with header.stamp adjusted by estimated offset
    """
    def __init__(self):
        super().__init__('compensation_node')
        self.declare_parameter('input_odom', 'odom_local')
        self.declare_parameter('output_odom', 'odom_synced')

        self.offset_ns = 0.0

        self.sub_status = self.create_subscription(Float32MultiArray, 'sync/status', self.on_status, 10)
        self.sub_odom = self.create_subscription(Odometry, self.get_parameter('input_odom').value, self.on_odom, 50)
        self.pub_odom = self.create_publisher(Odometry, self.get_parameter('output_odom').value, 50)

        log_dir = default_log_dir(self.get_name())
        self.logger_jsonl = JsonlLogger(f"{log_dir}/compensation_node.jsonl")
        self.get_logger().info(f"Logging to {self.logger_jsonl.path}")

    def on_status(self, msg: Float32MultiArray):
        if len(msg.data) >= 1:
            self.offset_ns = float(msg.data[0])

    def on_odom(self, msg: Odometry):
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose = msg.pose
        out.twist = msg.twist

        # Adjust: local_time - offset ~= master_time
        # If offset = slave - master, then master = slave - offset.
        t = out.header.stamp.sec * 1_000_000_000 + out.header.stamp.nanosec
        t_corr = int(t - self.offset_ns)
        out.header.stamp.sec = int(t_corr // 1_000_000_000)
        out.header.stamp.nanosec = int(t_corr % 1_000_000_000)

        self.pub_odom.publish(out)
        self.logger_jsonl.log({"event":"compensate", "t_in": int(t), "t_out": int(t_corr), "offset_ns": float(self.offset_ns)})

    def destroy_node(self):
        try:
            self.logger_jsonl.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = CompensationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
