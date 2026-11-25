from math import sqrt

import numpy as np
import pytest

from fusion import EKF, Noise, dead_reckoning, run, simulate, wrap


@pytest.fixture(scope="module")
def log0():
    return simulate(seed=0)


def rmse(X, log):
    return float(np.sqrt(np.mean(np.sum((X[:, :2] - log.truth[:, :2]) ** 2, 1))))


def test_jacobian_matches_finite_differences():
    e = EKF()
    e.x = np.array([3.0, -2.0, 0.7, 4.0, 0.02, 0.05, 1.03])
    gyro, accel, dt = 0.2, 0.3, 0.01
    F = e.F(gyro, dt)
    num = np.zeros((7, 7))
    for j in range(7):
        d = np.zeros(7)
        d[j] = 1e-6
        num[:, j] = (e.f(e.x + d, gyro, accel, dt) - e.f(e.x - d, gyro, accel, dt)) / 2e-6
    assert np.allclose(F, num, atol=1e-6)


def test_simulator_is_sane(log0):
    assert len(log0.t) == 12000 and len(log0.odo) == 2400
    t_gps = np.array([g[0] for g in log0.gps])
    assert not ((t_gps >= 45) & (t_gps < 65)).any()           # outage respected
    assert 0 < sum(g[3] for g in log0.gps) < 40                 # some multipath outliers


def test_ekf_beats_gps_and_dead_reckoning(log0):
    X, _, e = run(log0)
    g = np.array([(t, x, y) for t, x, y, o in log0.gps if not o])
    idx = np.round(g[:, 0] / 0.01).astype(int)
    gps_rmse = float(np.sqrt(np.mean(np.sum((g[:, 1:] - log0.truth[idx, :2]) ** 2, 1))))
    dr = dead_reckoning(log0)
    assert rmse(X, log0) < 0.5 * gps_rmse
    assert np.hypot(*(dr[-1, :2] - log0.truth[-1, :2])) > 50 * rmse(X, log0)


def test_learns_sensor_errors(log0):
    X, Pd, _ = run(log0)
    assert abs(X[-1, 4] - log0.bias[-1]) < 0.003                 # gyro bias, rad/s
    assert abs(X[-1, 5] - log0.spec.accel_bias) < 0.01           # accel bias, m/s^2
    assert abs(X[-1, 6] - log0.spec.odo_scale) < 0.005           # wheel scale
    assert np.sqrt(Pd[-1, 6]) < np.sqrt(Pd[0, 6]) / 5            # and becomes confident about it


def test_outlier_gating_precision_recall():
    tp = fp = fn = 0
    for seed in range(8):
        log = simulate(seed=seed)
        _, _, e = run(log)
        flags = np.array([o for *_, o in log.gps])
        acc = np.array(e.gps_decisions)
        tp += int(((~acc) & flags).sum())
        fp += int(((~acc) & ~flags).sum())
        fn += int((acc & flags).sum())
    assert fn == 0 and tp > 50
    assert fp / (len(flags) * 8) < 0.005  # ~0.1% expected from a 99.9% gate


def test_gating_matters(log0):
    X_g, _, _ = run(log0, EKF())
    X_n, _, _ = run(log0, EKF(gate=False))
    assert rmse(X_g, log0) < rmse(X_n, log0)


def test_outage_uncertainty_grows_then_collapses(log0):
    X, Pd, _ = run(log0)
    t = log0.t
    s = np.sqrt(Pd[:, 0] + Pd[:, 1])
    i0, i1, i2 = np.searchsorted(t, [44.9, 64.9, 70])
    assert s[i1] > 3 * s[i0]           # honest: uncertainty grows without GPS
    assert s[i2] < s[i1] / 3           # and shrinks once GPS is back
    err = np.hypot(*(X[:, :2] - log0.truth[:, :2]).T)
    assert err[i0:i1].max() < 5.0      # bounded drift over a 20 s outage


def test_odometry_helps_on_average():
    full, no = [], []
    for seed in range(6):
        log = simulate(seed=seed)
        full.append(rmse(run(log, EKF())[0], log))
        no.append(rmse(run(log, EKF(use_odo=False))[0], log))
    assert np.mean(full) < np.mean(no)


def test_filter_is_consistent_nees_and_nis():
    N = 12
    nees, nis = [], []
    for seed in range(N):
        log = simulate(seed=seed)
        X, _, e = run(log)
        d = X[:, :2] - log.truth[:, :2]
        nees.append(np.einsum("ni,nij,nj->n", d, np.linalg.inv(e.Pxy), d))
        nis += e.nis_gps
    avg = np.array(nees).mean(0)
    lo, hi = (2 * N - 1.96 * sqrt(4 * N)) / N, (2 * N + 1.96 * sqrt(4 * N)) / N
    assert 1.6 < avg[100:].mean() < 2.6                     # ideal 2 (2-D position)
    assert ((avg > lo) & (avg < hi)).mean() > 0.75
    assert 1.8 < np.mean(nis) < 2.2                         # innovations ~ chi2(2)


def test_covariance_stays_symmetric_positive_definite(log0):
    e = EKF()
    run(log0, e)
    assert np.allclose(e.P, e.P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(e.P) > 0)


def test_wrap():
    assert wrap(np.pi + 0.1) == pytest.approx(-np.pi + 0.1)
    assert wrap(-3 * np.pi) == pytest.approx(-np.pi)
