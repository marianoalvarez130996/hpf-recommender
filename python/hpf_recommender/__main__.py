import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from . import __version__, design
from .recommender import recommend


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hpf_recommender",
                                 description="Recomienda un filtro pasa-altos para cada pista.")
    ap.add_argument("inputs", nargs="+", help="Archivos de audio (usar siempre el original, sin filtrar)")
    ap.add_argument("--max_loss", type=float, default=1.0,
                    help="Pérdida máxima en la nota más grave, en dB (1.0)")
    ap.add_argument("--edge_loss", type=float, default=3.0,
                    help="Pérdida en el borde útil de una fuente sin altura si el ruido estacionario "
                         "no llega al borde (si llega, max_loss), en dB (3.0)")
    ap.add_argument("--range_db", type=float, default=30.0,
                    help="dB bajo el máximo espectral que todavía cuentan como contenido útil (30)")
    ap.add_argument("--fmin", type=float, default=27.5, help="f0 más grave buscado (27.5 Hz = A0)")
    ap.add_argument("--fmax", type=float, default=1000.0, help="f0 más agudo buscado")
    ap.add_argument("--order", type=int, default=None, help="Fija el orden, de 1 a 8 (si no, el mínimo que cumpla)")
    ap.add_argument("--max_order", type=int, default=8, help="Orden máximo automático (8 = 48 dB/oct)")
    ap.add_argument("--fc_max", type=float, default=300.0, help="Corte máximo recomendado")
    ap.add_argument("--no_snap", action="store_true", help="No redondear el corte a la serie R20")
    ap.add_argument("--export", action="store_true", help="Exportar el audio filtrado (mismo formato)")
    ap.add_argument("--plot", action="store_true", help="Guardar PNG con el espectro antes/después")
    ap.add_argument("--show", action="store_true", help="Abrir la gráfica en una ventana")
    ap.add_argument("--json", type=str, default="", help="Guardar el reporte en este archivo")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    reports, status = [], 0
    for name in args.inputs:
        path = Path(name)
        if not path.exists():
            print(f"No existe: {path}", file=sys.stderr)
            status = 1
            continue
        # un archivo ilegible o con NaN no corta el lote: se avisa y se sigue, como en C++
        try:
            info = sf.info(str(path))
            x, sr = sf.read(str(path), always_2d=True, dtype="float64")
            rec = recommend(x, sr, max_loss_db=args.max_loss, edge_loss_db=args.edge_loss,
                            range_db=args.range_db, fmin=args.fmin,
                            fmax=args.fmax, order=args.order, max_order=args.max_order,
                            fc_max=args.fc_max, snap=not args.no_snap)
        except (RuntimeError, ValueError, OSError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            status = 1
            continue
        report = {"file": path.name, "sr": sr, "channels": info.channels, **rec.as_dict()}
        reports.append(report)

        # sin nada medible que quitar se exporta solo el filtro fijo
        chain = rec.active and rec.cutoff_hz is not None
        if args.export and (chain or rec.base_hz):
            y = design.apply_recommendation(x, sr, rec)
            peak = float(np.max(np.abs(y)))
            if peak > 1.0 and info.subtype.startswith("PCM"):
                print(f"Aviso: pico de {20 * np.log10(peak):+.2f} dBFS en {path.name}; "
                      f"el {info.subtype} va a recortar.", file=sys.stderr)
            fc, n = (rec.cutoff_hz, rec.order) if chain else (rec.base_hz, rec.base_order)
            out = path.with_name(f"{path.stem}_HPF_{fc:g}Hz_ord{n}{path.suffix}")
            sf.write(str(out), y, sr, subtype=info.subtype, format=info.format)
            report["exported"] = str(out)

        if rec.cutoff_hz is not None and (args.plot or args.show):
            import matplotlib.pyplot as plt
            from .plot import before_after
            fig = before_after(rec, x, sr, path.name)
            if args.plot:
                out_png = path.with_name(path.stem + "_HPF_antes_despues.png")
                fig.savefig(out_png, dpi=150)
                report["plot"] = str(out_png)
            if args.show:
                plt.show()
            plt.close(fig)

    text = json.dumps(reports[0] if len(reports) == 1 else reports, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
