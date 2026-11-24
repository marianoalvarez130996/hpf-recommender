"""El recomendador sobre las pistas de una sesión. No guarda audio: solo el reporte y el espectro de
largo plazo de cada pista.

    python real_tracks.py "carpeta/con/wavs"      -> results/tracks/<pista>.json"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf

from _paths import RESULTS
from hpf_recommender import recommend


def one(path):
    path = Path(path)
    info = sf.info(str(path))
    x, sr = sf.read(str(path), always_2d=True, dtype="float64")
    r = recommend(x, sr)
    spec = r.spectrum
    out = {"file": path.name, "sr": sr, "subtype": info.subtype, "dur_s": info.frames / sr, "channels": info.channels,
           "report": r.as_dict(),
           "spectrum": {"centers": spec.centers.round(2).tolist(), "mean_db": spec.mean_db.round(2).tolist(),
                        "floor_db": spec.floor_db.round(2).tolist(),
                        "excess_re_max_db": (spec.excess_db - spec.excess_db.max()).round(2).tolist()}}
    (RESULTS / "tracks" / (path.stem + ".json")).write_text(json.dumps(out, ensure_ascii=False))
    return path.name


if __name__ == "__main__":
    (RESULTS / "tracks").mkdir(parents=True, exist_ok=True)
    files = sorted(Path(sys.argv[1]).glob("*.wav"))
    with Pool(6) as pool:
        for name in pool.imap_unordered(one, files):
            print(name, flush=True)
