import numpy as np
from scipy import signal

SR = 48000


def harmonic(f0, dur, sr=SR, n=12, fund_db=0.0, amp=0.3):
    t = np.arange(int(dur * sr)) / sr
    x = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t + k) for k in range(2, n + 1) if k * f0 < 0.45 * sr)
    x = x + 10 ** (fund_db / 20) * np.sin(2 * np.pi * f0 * t)
    return amp * x / np.max(np.abs(x))


def only_harmonics(f0, ks, dur, sr=SR, amp=0.3):
    t = np.arange(int(dur * sr)) / sr
    x = sum(np.sin(2 * np.pi * k * f0 * t + k) / k for k in ks)
    return amp * x / np.max(np.abs(x))


def bursts(dur, sr=SR, band=(200.0, 8000.0), every=0.25, decay=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    env = np.zeros(n)
    t = np.arange(int(5 * decay * sr)) / sr
    for start in range(0, n - len(t), int(every * sr)):
        env[start:start + len(t)] += np.exp(-t / decay)
    sos = signal.butter(4, band, btype="band", fs=sr, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(n)) * env
    return 0.3 * x / np.max(np.abs(x))


def rumble(dur, f, level, sr=SR, seed=1):
    """Ruido estacionario de banda angosta alrededor de f, con RMS = level."""
    rng = np.random.default_rng(seed)
    sos = signal.butter(2, [f / 1.3, f * 1.3], btype="band", fs=sr, output="sos")
    x = signal.sosfilt(sos, rng.standard_normal(int(dur * sr)))
    return level * x / np.sqrt(np.mean(x ** 2))


def hum60(n, level, sr=SR, ks=(1, 2, 3)):
    """Zumbido de red de 60 Hz con armónicos 1/k, con RMS = level."""
    t = np.arange(n) / sr
    x = sum(np.sin(2 * np.pi * 60.0 * k * t) / k for k in ks)
    return level * x / np.sqrt(np.mean(x ** 2))


def rare_low_note_over_hum(sr=SR):
    """G#1 que suena 0.3 s de 9.3 s, con la fundamental 17 dB abajo, entre notas más agudas y
    sobre hum de 60 Hz a -30 dB del RMS. Caso de la batería fuera de muestra (semilla 1)."""
    low = 51.91
    parts = [harmonic(low * 2 ** (k / 12), 0.75, sr=sr, fund_db=-17.0) for k in (14, 16, 3, 16, 10, 11, 13, 7, 19, 3, 7, 9)]
    parts.insert(6, harmonic(low, 0.3, sr=sr, fund_db=-17.0))
    x = np.concatenate(parts)
    return low, x + hum60(len(x), 10 ** (-30 / 20) * np.sqrt(np.mean(x ** 2)), sr)
