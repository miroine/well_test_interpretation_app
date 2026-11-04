import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from engine import reduce as R, models as M


def test_wavelet_denoise_reduces_noise():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 4000)
    clean = 3000 - 100 * (1 - np.exp(-t / 2))
    noisy = clean + rng.normal(0, 0.5, t.shape)
    den, sig = R.wavelet_denoise(noisy)
    assert abs(sig - 0.5) < 0.1
    assert np.sqrt(np.mean((den - clean) ** 2)) < 0.4 * np.sqrt(np.mean((noisy - clean) ** 2))


def test_wavelet_roundtrip_exact_without_threshold():
    y = np.random.default_rng(2).normal(size=256)
    den, _ = R.wavelet_denoise(y, strength=0.0)
    assert np.max(np.abs(den - y)) < 1e-9


def test_smart_reduce_keeps_shape_and_cuts_points():
    t = np.linspace(0.001, 300, 50000)
    p = 3500 - 150 * np.log10(1 + t)
    tr, pr, idx = R.smart_reduce(t, p, dp_tol=0.5, log_step=0.02)
    assert len(tr) < 0.1 * len(t)
    assert np.max(np.abs(np.interp(t, tr, pr) - p)) < 0.6


def test_detect_events_finds_rate_change():
    t = np.linspace(0.0, 100, 3000)
    p = np.where(t < 50, 3500 - 2.0 * t, 3400 + 4.0 * (t - 50))
    p = p + np.random.default_rng(1).normal(0, 0.05, t.shape)
    ev = R.detect_events(t, p)
    assert any(abs(e - 50) < 2.0 for e in ev)


def test_rate_reconstruction():
    P = M.WellTestParams(k=50, h=50, phi=0.18, mu=1, ct=1.5e-5, B=1.2, rw=0.354, pi=4000, C=0.01, S=2.0)
    sched = [(0.0, 1000.0), (40.0, 400.0), (90.0, 1500.0), (130.0, 0.0)]
    t = np.linspace(0.5, 200, 160)
    pwf = M.pwf_history(P, t, sched)
    edges = [0.0, 40.0, 90.0, 130.0, 200.0]
    q, rms = R.rate_from_pressure(P, t, pwf, edges, smooth=0.0)
    assert np.allclose(q, [1000, 400, 1500, 0], atol=60), q


def test_temperature_relaxation():
    t = np.linspace(0, 50, 200)
    T = 95 + (80 - 95) * np.exp(-t / 8) + np.random.default_rng(3).normal(0, 0.02, t.shape)
    fit, res = R.temperature_relaxation(t, T)
    assert abs(fit["tau"] - 8) < 0.5 and abs(fit["Tf"] - 95) < 0.1


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
