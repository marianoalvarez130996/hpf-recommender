# Compara lo que salió de REAPER con el núcleo.
#   <p>_1: plugin recién insertado (filtro apagado): debe ser el original por una ganancia fija.
#   <p>_2: con lo que aplicó el plugin: debe ser el original filtrado con el diseño del núcleo
#          (scipy.signal.butter, el mismo que hpf::design).
#   vst3_44k_1: render a 44.1 kHz; lo esperado es el núcleo sobre ese mismo render (lo que el plugin
#          oyó), deshecha la ganancia de la cadena medida a 48 kHz.
# También saca de cada .rpp el reporte que el plugin guardó en la sesión.
import base64
import html
import json
import plistlib
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

d = sys.argv[1].rstrip("/") + "/"
cli = sys.argv[2]
x, sr = sf.read(d + "gtr_line.wav", always_2d=True)
x = x[:, 0]
esperado = json.load(open(d + "esperado.json"))


def ajuste(ref, out, skip):
    n = min(len(ref), len(out))
    a, b = ref[skip:n], out[skip:n]
    g = float(np.dot(a, b) / np.dot(a, a))
    res = b - g * a
    return g, 10 * np.log10(np.sum(res ** 2) / np.sum(b ** 2))


def leer(nombre):
    try:
        y, s = sf.read(d + nombre + ".wav", always_2d=True)
    except Exception as e:  # noqa: BLE001
        print(f"{nombre}: no se pudo leer ({e})")
        return None, None
    print(f"{nombre}: {y.shape[0] / s:.2f} s, {s} Hz, {y.shape[1]} canal(es), pico {20 * np.log10(np.max(np.abs(y)) + 1e-30):.1f} dBFS")
    return y[:, 0], s


ganancia = None
for p in ("au", "vst3"):
    y1, _ = leer(p + "_1")
    if y1 is not None:
        g, e = ajuste(x, y1, 0)
        ganancia = ganancia or g
        print(f"  contra el original sin filtrar: ganancia {g:.4f} ({20 * np.log10(abs(g)):.2f} dB), residuo {e:.1f} dB")
    y2, _ = leer(p + "_2")
    if y2 is not None:
        sos = butter(esperado["order"], esperado["cutoff_hz"], "highpass", fs=sr, output="sos")
        g, e = ajuste(sosfilt(sos, x), y2, int(0.1 * sr))
        print(f"  contra el original filtrado por el núcleo ({esperado['cutoff_hz']} Hz, {esperado['slope_db_oct']} dB/oct): ganancia {g:.4f}, residuo {e:.1f} dB")
        g, e = ajuste(x, y2, int(0.1 * sr))
        print(f"  contra el original sin filtrar (para comparar): residuo {e:.1f} dB")

y44, s44 = leer("vst3_44k_1")
if y44 is not None and ganancia:
    sf.write(d + "oido_44k.wav", y44 / ganancia, s44, subtype="FLOAT")
    r = subprocess.run([cli, d + "oido_44k.wav", "--json", d + "esperado_44k.json"], capture_output=True, text=True)
    e44 = json.load(open(d + "esperado_44k.json"))
    print(f"  núcleo sobre lo que oyó el plugin a 44.1 kHz: corte {e44['cutoff_hz']} Hz, {e44['slope_db_oct']} dB/oct, filtra {e44['active']}")


def reporte(rpp, tipo):
    try:
        texto = open(d + rpp, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    bloque = re.search(r"<" + tipo + r" \"[^\"]*HPF![^\n]*\n(.*?)\n\s*>", texto, re.S)
    if not bloque:
        return "(no encontré el plugin)"
    datos = b""
    for linea in bloque.group(1).split("\n"):
        linea = linea.strip()
        if re.fullmatch(r"[A-Za-z0-9+/=]+", linea):
            try:
                datos += base64.b64decode(linea)
            except Exception:  # noqa: BLE001
                pass
    candidatos = [datos]
    i = datos.find(b"bplist00")
    if i >= 0:
        try:
            def recorrer(v):
                if isinstance(v, bytes):
                    candidatos.append(v)
                elif isinstance(v, dict):
                    for w in v.values():
                        recorrer(w)
                elif isinstance(v, list):
                    for w in v:
                        recorrer(w)
            recorrer(plistlib.loads(datos[i:]))
        except Exception:  # noqa: BLE001
            pass
    for m in re.finditer(rb"<data>\s*([A-Za-z0-9+/=\s]+?)\s*</data>", datos):
        try:
            candidatos.append(base64.b64decode(re.sub(rb"\s", b"", m.group(1))))
        except Exception:  # noqa: BLE001
            pass
    for c in candidatos:
        m = re.search(rb'summary="([^"]*)"', c)
        if m:
            return html.unescape(m.group(1).decode("utf-8", "replace"))
    return "(sin reporte en el estado)"


for rpp, tipo in (("prueba_au.rpp", "AU"), ("prueba_vst3.rpp", "VST"), ("prueba.rpp", "VST")):
    r = reporte(rpp, tipo)
    if r is not None:
        print(f"--- reporte guardado en {rpp}:\n{r}")
