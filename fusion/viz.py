"""Figures + GIF for the README:  python -m fusion.viz  ->  docs/"""
from __future__ import annotations

import io
import json
from math import sqrt
from pathlib import Path

import numpy as np

from .ekf import EKF, dead_reckoning, run
from .sim import simulate

TRUTH, GPS, BAD, DR, EKFC, OUT = "#1d3557", "#a8a8a8", "#e63946", "#f4a261", "#2a9d8f", "#ffe8cc"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
                         "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def ellipse(mean, cov, k=3.0, n=60):
    vals, vecs = np.linalg.eigh(cov)
    a = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(a), np.sin(a)])
    pts = vecs @ (np.sqrt(np.maximum(vals, 0))[:, None] * circle) * k
    return mean[0] + pts[0], mean[1] + pts[1]


def _gps(log):
    g = np.array([(t, x, y, o) for t, x, y, o in log.gps])
    return g[:, 0], g[:, 1:3], g[:, 3].astype(bool)


def fig_trajectory(out: Path, log, X, ekf):
    plt = _plt()
    dr = dead_reckoning(log)
    tg, gp, bad = _gps(log)
    o0, o1 = log.spec.gps_outage
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.scatter(gp[~bad, 0], gp[~bad, 1], s=7, color=GPS, label="GPS (σ = 1.5 m)", zorder=2)
    ax.scatter(gp[bad, 0], gp[bad, 1], s=40, marker="x", color=BAD, label="GPS multipath outliers (rejected)", zorder=3)
    ax.plot(log.truth[:, 0], log.truth[:, 1], color=TRUTH, lw=2.2, label="truth", zorder=4)
    m = (log.t < 30)
    ax.plot(dr[m, 0], dr[m, 1], color=DR, lw=1.8, ls="--", label="IMU dead reckoning (first 30 s)", zorder=4)
    ax.plot(X[:, 0], X[:, 1], color=EKFC, lw=1.8, label="EKF fused", zorder=5)
    out_m = (log.t >= o0) & (log.t < o1)
    ax.plot(log.truth[out_m, 0], log.truth[out_m, 1], color=OUT, lw=12, alpha=0.9, zorder=1, solid_capstyle="round",
            label=f"GPS outage ({o1 - o0:.0f} s)")
    for i in range(0, len(log.t), 250):
        ex, ey = ellipse(X[i, :2], ekf.Pxy[i])
        ax.plot(ex, ey, color=EKFC, lw=0.9, alpha=0.8, zorder=5)
    ax.set_aspect("equal")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="IMU + wheel odometry + GPS fused with an EKF (3σ ellipses every 2.5 s)")
    ax.legend(loc="best", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=105, facecolor="white")
    plt.close(fig)


def fig_errors(out: Path, log, X, Pd):
    plt = _plt()
    tg, gp, bad = _gps(log)
    idx = np.round(tg / (log.t[1] - log.t[0])).astype(int)
    gerr = np.hypot(*(gp - log.truth[idx, :2]).T)
    err = np.hypot(*(X[:, :2] - log.truth[:, :2]).T)
    sig = 3 * np.sqrt(Pd[:, 0] + Pd[:, 1])
    o0, o1 = log.spec.gps_outage
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axvspan(o0, o1, color=OUT, label="GPS outage")
    ax.scatter(tg[~bad], gerr[~bad], s=5, color=GPS, label=f"raw GPS (RMSE {np.sqrt(np.mean(gerr[~bad] ** 2)):.2f} m)")
    ax.fill_between(log.t, 0, sig, color=EKFC, alpha=0.15, label="EKF 3σ bound")
    ax.plot(log.t, err, color=EKFC, lw=1.8, label=f"EKF (RMSE {np.sqrt(np.mean(err ** 2)):.2f} m)")
    ax.set(xlabel="time (s)", ylabel="position error (m)", ylim=(0, max(8, sig.max() * 1.05)),
           title="Position error: the filter beats raw GPS, and stays inside its own 3σ bound")
    ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=105, facecolor="white")
    plt.close(fig)


def fig_biases(out: Path, log, X, Pd):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    items = [(4, log.bias, "gyro bias (deg/s)", np.degrees), (5, np.full_like(log.t, log.spec.accel_bias), "accel bias (m/s²)", lambda v: v),
             (6, np.full_like(log.t, log.spec.odo_scale), "wheel odometry scale", lambda v: v)]
    for ax, (j, true, label, conv) in zip(axes, items):
        s = 2 * np.sqrt(Pd[:, j])
        ax.fill_between(log.t, conv(X[:, j] - s), conv(X[:, j] + s), color=EKFC, alpha=0.18)
        ax.plot(log.t, conv(X[:, j]), color=EKFC, lw=1.8, label="estimate ± 2σ")
        ax.plot(log.t, conv(true), color=TRUTH, lw=1.5, ls="--", label="true")
        ax.set(title=label, xlabel="time (s)")
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Sensor errors the filter learns online", y=1.02, fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "biases.png", dpi=105, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_consistency(out: Path, n_runs: int = 20) -> dict:
    plt = _plt()
    nees, nis = [], []
    for seed in range(n_runs):
        log = simulate(seed=seed)
        X, _, e = run(log)
        d = X[:, :2] - log.truth[:, :2]
        nees.append(np.einsum("ni,nij,nj->n", d, np.linalg.inv(e.Pxy), d))
        nis += e.nis_gps
    avg = np.array(nees).mean(0)
    lo, hi = (2 * n_runs - 1.96 * sqrt(4 * n_runs)) / n_runs, (2 * n_runs + 1.96 * sqrt(4 * n_runs)) / n_runs
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axes[0]
    o0, o1 = log.spec.gps_outage
    ax.axvspan(o0, o1, color=OUT)
    ax.axhspan(lo, hi, color=EKFC, alpha=0.15, label="95% band")
    ax.axhline(2, color=TRUTH, ls="--", lw=1, label="ideal = 2")
    ax.plot(log.t, avg, color=EKFC, lw=1.4, label=f"average NEES over {n_runs} runs")
    ax.set(ylim=(0, 6), xlabel="time (s)", title="NEES: is the filter honest about its uncertainty?")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1]
    nis = np.array(nis)
    ax.hist(nis[nis < 14], bins=40, density=True, color=GPS, label="GPS innovations (NIS)")
    xs = np.linspace(0, 14, 200)
    ax.plot(xs, 0.5 * np.exp(-xs / 2), color=TRUTH, lw=2, label="χ²(2) expected")
    ax.axvline(13.816, color=BAD, ls="--", lw=1.2, label="99.9% gate")
    ax.set(xlabel="NIS", title=f"NIS matches χ²(2): mean {nis.mean():.2f} (ideal 2)")
    ax.legend(frameon=False, fontsize=8.5)
    for a in axes:
        a.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "consistency.png", dpi=105, facecolor="white")
    plt.close(fig)
    return {"avg_nees": round(float(avg[100:].mean()), 2), "nees_inside_95": round(float(((avg > lo) & (avg < hi)).mean()), 3),
            "avg_nis": round(float(nis.mean()), 2)}


def gif(out: Path, log, X, ekf):
    plt = _plt()
    from PIL import Image

    tg, gp, bad = _gps(log)
    o0, o1 = log.spec.gps_outage
    frames = []
    for i in range(int(35 / 0.01), int(80 / 0.01), 40):
        t = log.t[i]
        fig, ax = plt.subplots(figsize=(5.4, 5.4))
        c = log.truth[i, :2]
        ax.plot(log.truth[:i, 0], log.truth[:i, 1], color=TRUTH, lw=2)
        ax.plot(X[:i, 0], X[:i, 1], color=EKFC, lw=1.6)
        m = (tg > t - 6) & (tg <= t)
        ax.scatter(gp[m & ~bad, 0], gp[m & ~bad, 1], s=14, color=GPS)
        ax.scatter(gp[m & bad, 0], gp[m & bad, 1], s=50, marker="x", color=BAD)
        ex, ey = ellipse(X[i, :2], ekf.Pxy[i])
        ax.fill(ex, ey, color=EKFC, alpha=0.25)
        ax.plot(*X[i, :2], "o", color=EKFC, ms=7)
        ax.arrow(X[i, 0], X[i, 1], 4 * np.cos(X[i, 2]), 4 * np.sin(X[i, 2]), color=EKFC, width=0.25, head_width=1.2)
        ax.set_xlim(c[0] - 25, c[0] + 25)
        ax.set_ylim(c[1] - 25, c[1] + 25)
        ax.set_aspect("equal")
        status = "GPS OUTAGE: IMU + odometry only" if o0 <= t < o1 else "GPS + IMU + odometry"
        ax.set_title(f"t = {t:5.1f} s · {status}", loc="left", color=BAD if o0 <= t < o1 else "black")
        ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, dpi=68, facecolor="white")
        plt.close(fig)
        frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=48))
    frames[0].save(out / "ekf.gif", save_all=True, append_images=frames[1:], duration=70, loop=0, optimize=True)


def main() -> int:
    out = Path("docs")
    out.mkdir(exist_ok=True)
    log = simulate(seed=0)
    X, Pd, ekf = run(log)
    fig_trajectory(out, log, X, ekf)
    fig_errors(out, log, X, Pd)
    fig_biases(out, log, X, Pd)
    gif(out, log, X, ekf)
    stats = fig_consistency(out)
    err = np.hypot(*(X[:, :2] - log.truth[:, :2]).T)
    stats.update({"ekf_rmse_m": round(float(np.sqrt(np.mean(err ** 2))), 2), "rejected": ekf.rejected,
                  "true_outliers": int(sum(o for *_, o in log.gps))})
    print(json.dumps(stats, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
