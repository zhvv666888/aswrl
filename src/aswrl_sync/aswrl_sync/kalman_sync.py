import numpy as np

class KalmanSync:
    """Linear Kalman filter for clock sync.

    State x = [offset, delay, drift]^T
      offset: slave_time - master_time (ns)
      delay : one-way mean delay (ns)
      drift : offset drift rate (ns/s)
    Observation z = [offset_meas, delay_meas]^T derived from 4 timestamps (PTP-like).
    """
    def __init__(self, dt: float,
                 q_offset: float=1e6, q_delay: float=1e6, q_drift: float=1e2,
                 r_offset: float=1e7, r_delay: float=1e7):
        self.dt = float(dt)
        self.x = np.zeros((3, 1), dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64) * 1e12

        self.F = np.array([
            [1.0, 0.0, self.dt],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.Q = np.diag([q_offset, q_delay, q_drift]).astype(np.float64)

        self.H = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float64)

        self.R = np.diag([r_offset, r_delay]).astype(np.float64)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_offset_ns: float, z_delay_ns: float):
        z = np.array([[float(z_offset_ns)], [float(z_delay_ns)]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(3, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

    @property
    def offset_ns(self) -> float:
        return float(self.x[0, 0])

    @property
    def delay_ns(self) -> float:
        return float(self.x[1, 0])

    @property
    def drift_ns_per_s(self) -> float:
        return float(self.x[2, 0])
