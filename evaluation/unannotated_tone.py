"""El tono sin anotar de la línea 001 de IDMT-SMT-Bass-Single-Track: entre la G1 que termina en 11.011 s
y la C2 que empieza en 11.239 s suena un tono con fundamental cerca de 41 Hz que baja hasta cerca de
39.5 Hz, con sus armónicos. Imprime las notas anotadas alrededor y los picos de 25 a 200 Hz de cuatro
tramas de 120 ms (FFT de 2**18 puntos con ventana de Hann).

    python unannotated_tone.py ../../datos/idmt_bass"""
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

root = Path(sys.argv[1])
x, sr = sf.read(str(root / "audio" / "001.wav"))
for r in csv.reader(open(root / "misc" / "notes_csv" / "001_note_parameters.csv")):
    if 10.4 < float(r[0]) < 11.6:
        print(f"nota anotada: {float(r[0]):.3f}-{float(r[1]):.3f} s, MIDI {r[2]}, estilo {r[6]}")
for t0 in (10.9, 11.05, 11.15, 11.25):
    seg = x[int(t0 * sr):int((t0 + 0.12) * sr)]
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 18))
    f = np.fft.rfftfreq(1 << 18, 1 / sr)
    sel = (f > 25) & (f < 200)
    peaks = signal.find_peaks(20 * np.log10(mag[sel] + 1e-12), prominence=8)[0]
    top = sorted(sorted(peaks, key=lambda i: -mag[sel][i])[:4])
    print(f"{t0:.2f} s:", ", ".join(f"{f[sel][i]:.1f} Hz ({20 * np.log10(mag[sel][i] / mag.max()):.1f} dB)" for i in top))
