"""Simulated ground robot with realistic sensors: a drifting IMU, slipping wheel odometry and noisy GPS
with an outage and multipath outliers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorSpec:
    imu_hz: float = 100.0
    odo_hz: float = 20.0
    gps_hz: float = 5.0
    gyro_noise: float = 0.01          # rad/s white noise (per sample)
    gyro_bias0: float = 0.02          # rad/s initial bias (~1.1 deg/s, a cheap MEMS gyro)
    gyro_bias_walk: float = 0.0005    # rad/s/sqrt(s) bias random walk
    accel_noise: float = 0.05         # m/s^2
    accel_bias: float = 0.06          # m/s^2 constant offset (typical consumer MEMS)
    odo_noise: float = 0.05           # m/s
    odo_scale: float = 1.03           # wheel radius 3% off (tyre pressure, wear)
    gps_noise: float = 1.5            # m (1 sigma per axis)
    gps_outage: tuple[float, float] = (45.0, 65.0)   # seconds without GPS (tunnel, under trees)
    gps_outlier_rate: float = 0.03    # multipath jumps
    gps_outlier_size: float = 15.0    # m


@dataclass
class Log:
    t: np.ndarray
    truth: np.ndarray                 # [N, 4] x, y, theta, v at IMU rate
    gyro: np.ndarray
    accel: np.ndarray
    bias: np.ndarray                  # true gyro bias
    odo: list = field(default_factory=list)   # (t, speed)
    gps: list = field(default_factory=list)   # (t, x, y, is_outlier)
    spec: SensorSpec = field(default_factory=SensorSpec)


def trajectory(duration: float = 120.0, dt: float = 0.01):
    """A smooth figure-of-eight-ish drive with speed changes: v(t), omega(t)."""
    t = np.arange(0, duration, dt)
    v = 4.0 + 1.5 * np.sin(2 * np.pi * t / 40) + 0.8 * np.sin(2 * np.pi * t / 13)
    v = np.clip(v, 0.5, None) * np.minimum(1.0, t / 3.0)  # gentle start
    w = 0.18 * np.sin(2 * np.pi * t / 30) + 0.06 * np.sin(2 * np.pi * t / 7.5)
    th = np.cumsum(w) * dt
    x = np.cumsum(v * np.cos(th)) * dt
    y = np.cumsum(v * np.sin(th)) * dt
    return t, np.column_stack([x, y, th, v]), w


def simulate(duration: float = 120.0, spec: SensorSpec | None = None, seed: int = 0) -> Log:
    spec = spec or SensorSpec()
    rng = np.random.default_rng(seed)
    dt = 1.0 / spec.imu_hz
    t, truth, w = trajectory(duration, dt)
    accel_true = np.gradient(truth[:, 3], dt)
    bias = spec.gyro_bias0 + np.cumsum(rng.normal(0, spec.gyro_bias_walk * np.sqrt(dt), len(t)))
    gyro = w + bias + rng.normal(0, spec.gyro_noise, len(t))
    accel = accel_true + spec.accel_bias + rng.normal(0, spec.accel_noise, len(t))
    log = Log(t, truth, gyro, accel, bias, spec=spec)
    odo_every, gps_every = int(spec.imu_hz / spec.odo_hz), int(spec.imu_hz / spec.gps_hz)
    for i in range(len(t)):
        if i % odo_every == 0:
            log.odo.append((t[i], spec.odo_scale * truth[i, 3] + rng.normal(0, spec.odo_noise)))
        if i % gps_every == 0 and not (spec.gps_outage[0] <= t[i] < spec.gps_outage[1]):
            gx, gy = truth[i, :2] + rng.normal(0, spec.gps_noise, 2)
            outlier = rng.random() < spec.gps_outlier_rate
            if outlier:
                a = rng.uniform(0, 2 * np.pi)
                gx, gy = gx + spec.gps_outlier_size * np.cos(a), gy + spec.gps_outlier_size * np.sin(a)
            log.gps.append((t[i], gx, gy, outlier))
    return log
