import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

from .clock_drift import SimClock

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

class DummyOdom(Node):
    """Publish simple circular motion as odometry with drifted timestamps.

    Use for quick experiments when you don't have real robot state from Gazebo plugins.
    """
    def __init__(self):
        super().__init__('dummy_odom')
        self.declare_parameter('topic', 'odom_local')
        self.declare_parameter('radius', 1.0)
        self.declare_parameter('omega', 0.2)
        self.declare_parameter('clock_offset_ns', 0)
        self.declare_parameter('clock_skew_ppm', 0.0)

        self.topic = str(self.get_parameter('topic').value)
        self.pub = self.create_publisher(Odometry, self.topic, 10)

        self.clock_model = SimClock(
            offset_ns=int(self.get_parameter('clock_offset_ns').value),
            skew_ppm=float(self.get_parameter('clock_skew_ppm').value),
            noise_ns=0,
            seed=999
        )

        self.t0 = self.get_clock().now().nanoseconds
        self.timer = self.create_timer(0.05, self.tick)

    def tick(self):
        now_true = self.get_clock().now().nanoseconds
        t = (now_true - self.t0) / 1e9
        r = float(self.get_parameter('radius').value)
        w = float(self.get_parameter('omega').value)
        x = r * math.cos(w*t)
        y = r * math.sin(w*t)
        yaw = w*t

        msg = Odometry()
        # drifted timestamp
        local = self.clock_model.local_time_ns(now_true)
        msg.header.stamp.sec = int(local // 1_000_000_000)
        msg.header.stamp.nanosec = int(local % 1_000_000_000)
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation = yaw_to_quat(yaw)
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = DummyOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
