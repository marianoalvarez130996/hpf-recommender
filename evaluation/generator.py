"""Generador de la batería aleatoria, compartido por battery.py y baselines.py.

Líneas tonales: nota más grave entre E1 y E3, que suena entre 2 y 15 % del tiempo, fundamental entre
0 y -20 dB bajo una serie armónica 1/k y notas de 2 a 19 semitonos más arriba. Percusión: ruido en
ráfagas con borde inferior entre 80 y 400 Hz. A ambas se suma rumble de banda angosta y, en 30 % de
los casos, hum de 60 Hz (contaminate, que también usa real_annotated.py). Con la misma semilla, las
señales son idénticas."""
import numpy as np
from scipy import signal

SR = 48000


def tone(f0, n, fund_db, phase):
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * k * f0 * t + phase[k]) / k for k in range(2, 13) if k * f0 < 0.45 * SR)
    return x + 10 ** (fund_db / 20) * np.sin(2 * np.pi * f0 * t + phase[1])


def fade(x, ms=10):
    m = int(ms * SR / 1000)
    w = np.ones(len(x)); w[:m] = np.linspace(0, 1, m); w[-m:] = np.linspace(1, 0, m)
    return x * w


def hum(n, rms, sr=SR):
    t = np.arange(n) / sr
    y = sum(np.sin(2 * np.pi * 60 * k * t) / k for k in (1, 2, 3))
    return rms * y / np.sqrt(np.mean(y ** 2))


def narrow_noise(rng, n, f, rms, sr=SR):
    sos = signal.butter(2, [f / 1.3, f * 1.3], btype='band', fs=sr, output='sos')
    y = signal.sosfilt(sos, rng.standard_normal(n))
    return rms * y / np.sqrt(np.mean(y ** 2))


def contaminate(rng, x, f_hi, r_db_range, sr=SR):
    """Suma a x rumble de banda angosta centrado entre 15 Hz y f_hi, con nivel en r_db_range dB
    respecto del valor eficaz de x, y en 30 % de los casos hum de 60 Hz entre -40 y -25 dB.
    Devuelve (señal, rumble, f_r, r_db, hum)."""
    rms = np.sqrt(np.mean(x ** 2))
    f_r = rng.uniform(15.0, f_hi)
    r_db = rng.uniform(*r_db_range)
    r = narrow_noise(rng, len(x), f_r, rms * 10 ** (r_db / 20), sr)
    y = x + r
    has_hum = rng.random() < 0.3
    if has_hum:
        y = y + hum(len(x), rms * 10 ** (rng.uniform(-40, -25) / 20), sr)
    return y, r, f_r, r_db, has_hum


class Battery:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def case(self, i, n_tonal):
        """El caso i: tonal si i < n_tonal, percusivo si no. Devuelve (señal, rumble, datos, fuente
        limpia), donde la fuente limpia es la señal sin el rumble ni el hum."""
        return self.tonal_case() if i < n_tonal else self.perc_case()

    def tonal_case(self, dur=10.0):
        rng = self.rng
        low = 41.2034 * 2 ** (rng.integers(0, 25) / 12)          # E1 .. E3
        share = rng.uniform(0.02, 0.15)                           # tiempo que suena la nota más grave
        fund_db = rng.uniform(-20, 0)
        n = int(dur * SR)
        n_low = int(share * n)
        notes, used = [], 0
        while used < n - n_low:
            k = int(rng.integers(2, 20))
            m = min(int(rng.uniform(0.25, 0.8) * SR), n - n_low - used)
            if m < int(0.1 * SR):
                break
            notes.append(fade(tone(low * 2 ** (k / 12), m, fund_db, rng.uniform(0, 2 * np.pi, 13))))
            used += m
        pos = int(rng.integers(0, len(notes) + 1))
        notes.insert(pos, fade(tone(low, n_low, fund_db, rng.uniform(0, 2 * np.pi, 13))))
        x = np.concatenate(notes)
        clean = 0.3 * x / np.max(np.abs(x))
        x, r, f_r, r_db, has_hum = contaminate(rng, clean, 0.6 * low, (-35, -10))
        return x, r, dict(kind='tonal', low=low, share=share, fund_db=fund_db, f_r=f_r, r_db=r_db, hum=has_hum), clean

    def perc_case(self, dur=10.0):
        rng = self.rng
        f_b = rng.uniform(80, 400)
        n = int(dur * SR)
        env = np.zeros(n)
        t = np.arange(int(0.25 * SR)) / SR
        every = rng.uniform(0.15, 0.5)
        for s in range(0, n - len(t), int(every * SR)):
            env[s:s + len(t)] += np.exp(-t / 0.05) * rng.uniform(0.5, 1.0)
        sos = signal.butter(4, [f_b, 8000.0], btype='band', fs=SR, output='sos')
        x = signal.sosfilt(sos, rng.standard_normal(n)) * env
        clean = 0.3 * x / np.max(np.abs(x))
        x, r, f_r, r_db, has_hum = contaminate(rng, clean, 0.5 * f_b, (-30, -5))
        return x, r, dict(kind='perc', low=f_b, f_r=f_r, r_db=r_db, hum=has_hum), clean
