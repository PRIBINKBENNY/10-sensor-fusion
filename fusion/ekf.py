"""Extended Kalman filter fusing IMU (prediction), wheel odometry and GPS (updates).

State x = [px, py, theta, v, b_g, b_a, k_odo]
  px, py   position (m)            theta  heading (rad)        v  forward speed (m/s)
  b_g      gyro bias (rad/s)       b_a    accelerometer bias (m/s^2)
  k_odo    odometry scale factor (wheel radius error)
Prediction (100 Hz):  theta += (gyro - b_g) dt;  v += (accel - b_a) dt;  p += v [cos, sin] dt
Updates: odometry z = k_odo * v (20 Hz); GPS z = [px, py] (5 Hz) with chi-square outlier gating.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CHI2_2DOF_999 = 13.816   # 99.9% gate for a 2-D measurement
CHI2_1DOF_999 = 10.828


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Noise:
    gyro: float = 0.01
    accel: float = 0.05
    bias_walk: float = 0.0005
    accel_bias_walk: float = 0.001
    scale_walk: float = 1e-5
    odo: float = 0.05
    gps: float = 1.5


@dataclass
class EKF:
    x: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0, 0, 0, 1.0]))
    P: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.05, 0.1, 0.03 ** 2, 0.1 ** 2, 0.05 ** 2]))
    noise: Noise = field(default_factory=Noise)
    use_gps: bool = True
    use_odo: bool = True
    gate: bool = True
    nis_gps: list = field(default_factory=list)
    Pxy: np.ndarray | None = None
    rejected: int = 0
    gps_decisions: list = field(default_factory=list)  # (accepted?) per GPS fix, for precision/recall

    def F(self, gyro: float, dt: float) -> np.ndarray:
        """Jacobian of the motion model w.r.t. the state."""
        _, _, th, v = self.x[:4]
        F = np.eye(7)
        F[0, 2] = -v * np.sin(th) * dt
        F[0, 3] = np.cos(th) * dt
        F[1, 2] = v * np.cos(th) * dt
        F[1, 3] = np.sin(th) * dt
        F[2, 4] = -dt
        F[3, 5] = -dt
        return F

    def f(self, x, gyro, accel, dt):
        px, py, th, v, bg, ba, k = x
        th2 = wrap(th + (gyro - bg) * dt)
        return np.array([px + v * np.cos(th) * dt, py + v * np.sin(th) * dt, th2, v + (accel - ba) * dt, bg, ba, k])

    def predict(self, gyro: float, accel: float, dt: float) -> None:
        F = self.F(gyro, dt)
        self.x = self.f(self.x, gyro, accel, dt)
        n = self.noise
        Q = np.diag([0, 0, (n.gyro * dt) ** 2, (n.accel * dt) ** 2, n.bias_walk ** 2 * dt,
                     n.accel_bias_walk ** 2 * dt, n.scale_walk ** 2 * dt])
        Q[0, 0] = Q[1, 1] = (0.01 * dt) ** 2  # tiny process noise on position (unmodelled slip)
        self.P = F @ self.P @ F.T + Q

    def _update(self, z, h, H, R, gate_thr=None) -> float | None:
        y = z - h
        S = H @ self.P @ H.T + R
        Si = np.linalg.inv(S)
        nis = float(y @ Si @ y)
        if gate_thr is not None and self.gate and nis > gate_thr:
            self.rejected += 1
            return None
        K = self.P @ H.T @ Si
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(7) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form: stays symmetric positive-definite
        return nis

    def update_gps(self, px: float, py: float) -> float | None:
        if not self.use_gps:
            return None
        H = np.zeros((2, 7))
        H[0, 0] = H[1, 1] = 1
        nis = self._update(np.array([px, py]), self.x[:2], H, np.eye(2) * self.noise.gps ** 2, CHI2_2DOF_999)
        self.gps_decisions.append(nis is not None)
        if nis is not None:
            self.nis_gps.append(nis)
        return nis

    def update_odo(self, speed: float) -> float | None:
        if not self.use_odo:
            return None
        v, k = self.x[3], self.x[6]
        H = np.zeros((1, 7))
        H[0, 3], H[0, 6] = k, v
        return self._update(np.array([speed]), np.array([k * v]), H, np.array([[self.noise.odo ** 2]]), CHI2_1DOF_999)


def run(log, ekf: EKF | None = None, init_from_truth: bool = True):
    """Replay a sensor log through the filter. Returns states [N,7] and covariance diagonals [N,7]."""
    ekf = ekf or EKF()
    if init_from_truth:  # known start pose (e.g. a charging dock); bias and scale unknown
        ekf.x[:4] = log.truth[0]
    dt = log.t[1] - log.t[0]
    odo = iter(log.odo)
    gps = iter(log.gps)
    next_odo, next_gps = next(odo, None), next(gps, None)
    X, Pd = np.zeros((len(log.t), 7)), np.zeros((len(log.t), 7))
    ekf.Pxy = np.zeros((len(log.t), 2, 2))  # full position covariance, for NEES and ellipses
    for i, t in enumerate(log.t):
        if i > 0:
            ekf.predict(log.gyro[i - 1], log.accel[i - 1], dt)
        while next_odo is not None and next_odo[0] <= t + 1e-9:
            ekf.update_odo(next_odo[1])
            next_odo = next(odo, None)
        while next_gps is not None and next_gps[0] <= t + 1e-9:
            ekf.update_gps(next_gps[1], next_gps[2])
            next_gps = next(gps, None)
        X[i], Pd[i], ekf.Pxy[i] = ekf.x, np.diag(ekf.P), ekf.P[:2, :2]
    return X, Pd, ekf


def dead_reckoning(log) -> np.ndarray:
    """IMU-only integration from the true start: what you get without corrections."""
    dt = log.t[1] - log.t[0]
    x = np.zeros((len(log.t), 4))
    x[0] = log.truth[0]
    for i in range(1, len(log.t)):
        px, py, th, v = x[i - 1]
        x[i] = [px + v * np.cos(th) * dt, py + v * np.sin(th) * dt, wrap(th + log.gyro[i - 1] * dt), v + log.accel[i - 1] * dt]
    return x
