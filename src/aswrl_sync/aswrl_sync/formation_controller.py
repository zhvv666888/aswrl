import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from .logging_utils import JsonlLogger, default_log_dir

class FormationController(Node):
    """A minimal formation error evaluator.

    This node does NOT drive Gazebo physics. It computes formation error between robots using
    their odometry topics and publishes an error metric for logging/analysis.

    Expected positions (relative to dog1):
      dog2 target: (+1.0m, 0.0m)
      uav1 target: (0.0m, +1.0m)
    """
    def __init__(self):
        super().__init__('formation_controller')
        self.declare_parameter('dog1_odom', '/dog1/odom_local')
        self.declare_parameter('dog2_odom', '/dog2/odom_synced')
        self.declare_parameter('uav1_odom', '/uav1/odom_synced')

        self.p1 = None
        self.p2 = None
        self.p3 = None

        self.create_subscription(Odometry, self.get_parameter('dog1_odom').value, self.cb1, 50)
        self.create_subscription(Odometry, self.get_parameter('dog2_odom').value, self.cb2, 50)
        self.create_subscription(Odometry, self.get_parameter('uav1_odom').value, self.cb3, 50)

        self.pub_err = self.create_publisher(Float32MultiArray, 'formation/error', 10)
        self.timer = self.create_timer(0.1, self.tick)

        log_dir = default_log_dir(self.get_name())
        self.logger_jsonl = JsonlLogger(f"{log_dir}/formation_controller.jsonl")
        self.get_logger().info(f"Logging to {self.logger_jsonl.path}")

    def cb1(self, msg: Odometry):
        self.p1 = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def cb2(self, msg: Odometry):
        self.p2 = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def cb3(self, msg: Odometry):
        self.p3 = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def tick(self):
        if self.p1 is None or self.p2 is None or self.p3 is None:
            return
        x1,y1 = self.p1
        x2,y2 = self.p2
        x3,y3 = self.p3

        # desired
        dx2, dy2 = 1.0, 0.0
        dx3, dy3 = 0.0, 1.0

        e2 = math.hypot((x2-x1)-dx2, (y2-y1)-dy2)
        e3 = math.hypot((x3-x1)-dx3, (y3-y1)-dy3)
        rmse = math.sqrt(0.5*(e2*e2 + e3*e3))

        msg = Float32MultiArray()
        msg.data = [float(e2), float(e3), float(rmse)]
        self.pub_err.publish(msg)

        self.logger_jsonl.log({"event":"formation_error", "e2": e2, "e3": e3, "rmse": rmse})

    def destroy_node(self):
        try:
            self.logger_jsonl.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = FormationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
