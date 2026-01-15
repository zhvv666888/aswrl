import os
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int64MultiArray, Float32MultiArray

from .clock_drift import SimClock
from .kalman_sync import KalmanSync
from .window_controller import WindowSmoother, FixedWindowController, RuleASWController, RLASWController
from .logging_utils import JsonlLogger, default_log_dir

class SyncClient(Node):
    def __init__(self):
        super().__init__('sync_client')

        # basic params
        self.declare_parameter('sync_dt', 0.1)
        self.declare_parameter('strategy', 'baseline_fw')  # nocomp | baseline_fw | baseline_rule | asw_rl
        self.declare_parameter('fixed_window', 16)
        self.declare_parameter('policy_path', '')
        self.declare_parameter('clock_offset_ns', 5_000_000)  # 5ms
        self.declare_parameter('clock_skew_ppm', 25.0)
        self.declare_parameter('clock_noise_ns', 2000)

        self.dt = float(self.get_parameter('sync_dt').value)
        self.strategy = str(self.get_parameter('strategy').value)

        # drift model
        self.clock_model = SimClock(
            offset_ns=int(self.get_parameter('clock_offset_ns').value),
            skew_ppm=float(self.get_parameter('clock_skew_ppm').value),
            noise_ns=int(self.get_parameter('clock_noise_ns').value),
            seed=456
        )

        # KF and window
        self.kf = KalmanSync(dt=self.dt)
        self.smoother_off = WindowSmoother(W=int(self.get_parameter('fixed_window').value))
        self.smoother_del = WindowSmoother(W=int(self.get_parameter('fixed_window').value))

        # strategy controllers
        self.fixed_ctrl = FixedWindowController(int(self.get_parameter('fixed_window').value))
        self.rule_ctrl = RuleASWController()

        self.rl_ctrl = None
        if self.strategy == 'asw_rl':
            import torch
            p = str(self.get_parameter('policy_path').value)
            if not p or not os.path.exists(p):
                self.get_logger().warn("strategy=asw_rl but policy_path missing. Fallback to rule baseline.")
                self.strategy = 'baseline_rule'
            else:
                policy = torch.jit.load(p) if p.endswith('.pt') else torch.load(p, map_location='cpu')
                policy.eval()
                self.rl_ctrl = RLASWController(policy)

        # ROS topics
        self.sub_req = self.create_subscription(Int64MultiArray, 'sync/request', self.on_req, 100)
        self.sub_follow = self.create_subscription(Int64MultiArray, 'sync/followup', self.on_follow, 100)
        self.pub_resp = self.create_publisher(Int64MultiArray, 'sync/response', 10)

        # publish status: [offset_est_ns, delay_est_ns, drift_est, window, sync_err_ns]
        self.pub_status = self.create_publisher(Float32MultiArray, 'sync/status', 10)

        self.pending = {}  # seq -> (t1, t2, t3)

        # log
        log_dir = default_log_dir(self.get_name())
        self.logger_jsonl = JsonlLogger(f"{log_dir}/sync_client.jsonl")
        self.get_logger().info(f"Logging to {self.logger_jsonl.path}")

        self.last_sync_err = 0.0

    def now_true_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def now_local_ns(self) -> int:
        return self.clock_model.local_time_ns(self.now_true_ns())

    def on_req(self, msg: Int64MultiArray):
        # request: [seq, t1]
        if len(msg.data) < 2:
            return
        seq, t1 = int(msg.data[0]), int(msg.data[1])
        t2 = self.now_local_ns()  # receive timestamp (local)
        t3 = self.now_local_ns()  # send timestamp (local) - simplified
        self.pending[seq] = (t1, t2, t3)

        resp = Int64MultiArray()
        resp.data = [seq, t1, t2, t3]
        self.pub_resp.publish(resp)

        self.logger_jsonl.log({"event":"recv_req_send_resp", "seq": seq, "t1": t1, "t2": t2, "t3": t3})

    def on_follow(self, msg: Int64MultiArray):
        # followup: [seq, t4]
        if len(msg.data) < 2:
            return
        seq, t4 = int(msg.data[0]), int(msg.data[1])
        if seq not in self.pending:
            return
        t1, t2, t3 = self.pending.pop(seq)

        # PTP-like measurement (in local/master mixed timestamps)
        # m1 = t2 - t1 ; m2 = t4 - t3 ; offset = (m1 - m2)/2 ; delay = (m1 + m2)/2
        m1 = (t2 - t1)
        m2 = (t4 - t3)
        offset_meas = 0.5 * (m1 - m2)
        delay_meas = 0.5 * (m1 + m2)

        # KF update
        self.kf.predict()
        self.kf.update(offset_meas, delay_meas)

        # smoothing with window W (possibly adaptive)
        sync_err = self.kf.offset_ns  # using offset as sync error proxy
        delay_var = self.smoother_del.var()

        W = self.select_window(delay_var=delay_var, sync_err_ns=sync_err)
        if W != self.smoother_off.W:
            self.smoother_off.set_W(W)
            self.smoother_del.set_W(W)

        off_s = self.smoother_off.add(self.kf.offset_ns)
        del_s = self.smoother_del.add(self.kf.delay_ns)

        # publish status
        st = Float32MultiArray()
        st.data = [float(off_s), float(del_s), float(self.kf.drift_ns_per_s), float(self.smoother_off.W), float(sync_err)]
        self.pub_status.publish(st)

        self.logger_jsonl.log({
            "event":"update",
            "seq": seq,
            "offset_meas": float(offset_meas),
            "delay_meas": float(delay_meas),
            "offset_est": float(self.kf.offset_ns),
            "delay_est": float(self.kf.delay_ns),
            "drift_est": float(self.kf.drift_ns_per_s),
            "W": int(self.smoother_off.W),
            "sync_err": float(sync_err),
        })

        self.last_sync_err = float(sync_err)

    def select_window(self, delay_var: float, sync_err_ns: float) -> int:
        if self.strategy == 'nocomp':
            return int(self.smoother_off.W)
        if self.strategy == 'baseline_fw':
            return int(self.fixed_ctrl.choose_W())
        if self.strategy == 'baseline_rule':
            return int(self.rule_ctrl.choose_W(delay_var=delay_var, sync_err_ns=sync_err_ns))
        if self.strategy == 'asw_rl' and self.rl_ctrl is not None:
            # state vector: [delay_mean, delay_var, loss_rate(placeholder), sync_err, delta_err, window_norm]
            delay_mean = float(self.kf.delay_ns)
            loss_rate = 0.0
            delta_err = float(sync_err_ns - self.last_sync_err)
            window_norm = float(self.smoother_off.W) / 64.0
            state = np.array([
                delay_mean / 1e9,
                delay_var / 1e18,
                loss_rate,
                sync_err_ns / 1e9,
                delta_err / 1e9,
                window_norm,
            ], dtype=np.float32)
            return int(self.rl_ctrl.choose_W(state))
        return int(self.smoother_off.W)

    def destroy_node(self):
        try:
            self.logger_jsonl.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = SyncClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
