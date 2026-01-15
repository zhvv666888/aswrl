import random

class SimClock:
    """Software clock drift/offset injector.

    We treat `true_time_ns` as the reference time (e.g., ROS /clock or wall time),
    and generate a local clock reading with offset + skew (ppm drift) + optional noise.
    """
    def __init__(self, offset_ns: int = 0, skew_ppm: float = 0.0, noise_ns: int = 0, seed: int = 0):
        self.offset_ns = int(offset_ns)
        self.skew = 1.0 + float(skew_ppm) * 1e-6
        self.noise_ns = int(noise_ns)
        self._rnd = random.Random(seed)

    def local_time_ns(self, true_time_ns: int) -> int:
        n = self._rnd.randint(-self.noise_ns, self.noise_ns) if self.noise_ns > 0 else 0
        return int(true_time_ns * self.skew + self.offset_ns + n)
