"""Las cifras del paper a partir de results/. Escribe results/summary.json.

    python summarize.py

Acierto de la nota más grave: a menos de 50 cents de la real (sin nota = fallo). Intervalos de Wilson
al 95 %. El método completo y la ablación (sin la prueba por trama) se comparan sobre los mismos
casos, así que la prueba es McNemar exacta.

Semillas: 2026 se usó al desarrollar el método; 1 a 20, al diagnosticar sus fallos, elegir la prueba
por trama y los grados de libertad, ajustar la versión 5.2 y medir el filtro fijo de 20 Hz; 21 a 30
fueron el conjunto de prueba de una configuración anterior y ya no se usan; 31 a 40 se generaron una
sola vez, con el código ya congelado (results/code_frozen_sha256.txt), y son el conjunto de prueba."""
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from _paths import RESULTS
from methods import METHODS

DEV, DESIGN, TEST = [2026], list(range(1, 21)), list(range(31, 41))


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar(a, b):
    """Casos que a acierta y b no, al revés, y el valor p exacto de dos colas."""
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    n = a_only + b_only
    return a_only, b_only, (1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(a_only, b_only), n, 0.5)))


def hit(f0, low):
    return f0 is not None and abs(1200 * math.log2(f0 / low)) < 50


def accuracy(hits):
    lo, hi = wilson(sum(hits), len(hits))
    return {"k": sum(hits), "pct": 100 * sum(hits) / len(hits), "wilson": [100 * lo, 100 * hi]}


def loss_stats(loss):
    loss = np.array(loss)
    return {"median": float(np.median(loss)), "max": float(loss.max()), "n_over_1_05": int((loss > 1.05).sum())}


def battery(seeds):
    rows = [r for s in seeds for r in json.load(open(RESULTS / f"battery_{s}.json"))["cases"]]
    ton, per = [r for r in rows if r["kind"] == "tonal"], [r for r in rows if r["kind"] == "perc"]
    full, abl = [hit(r["f0"], r["low"]) for r in ton], [hit(r["ablation_f0"], r["low"]) for r in ton]
    miss = [r for r in ton if not hit(r["f0"], r["low"])]
    e = np.array([r["rej_energy"] for r in per])
    hum, no = [r for r in rows if r["hum"]], [r for r in rows if not r["hum"]]
    return {
        "n_tonal": len(ton), "n_perc": len(per),
        "acc": accuracy(full), "acc_ablation": accuracy(abl),
        "mcnemar_ablation": dict(zip(("frame_test_worse", "frame_test_better", "p"), mcnemar(abl, full))),
        "loss": loss_stats([r["loss"] for r in ton]), "loss_ablation": loss_stats([r["ablation_loss"] for r in ton]),
        "tonal_rej": {"proxy_median": float(np.median([r["rej"] for r in ton])),
                      "energy_median": float(np.median([r["rej_energy"] for r in ton])),
                      "energy_p5": float(np.percentile([r["rej_energy"] for r in ton], 5))},
        "perc": {"loss_median": float(np.median([r["loss"] for r in per])),
                 "loss_max": float(np.max([r["loss"] for r in per])),
                 "rej_proxy_median": float(np.median([r["rej"] for r in per])),
                 "rej_energy_median": float(np.median(e)), "n_under_6db": int((e < 6).sum())},
        "hum": {"detected": sum(r["hum_flagged"] for r in hum), "with_hum": len(hum),
                "false_alarms": sum(r["hum_flagged"] for r in no), "without_hum": len(no)},
        "misses": {"n": len(miss), "lower": sum(r["f0"] is not None and r["f0"] < r["low"] for r in miss),
                   "higher": sum(r["f0"] is not None and r["f0"] > r["low"] for r in miss),
                   "none": sum(r["f0"] is None for r in miss),
                   "loss_median": float(np.median([r["loss"] for r in miss])) if miss else 0.0},
        "failures": [{"low": round(r["low"], 1), "share": round(r["share"], 3), "fund_db": round(r["fund_db"], 1),
                      "hum": r["hum"], "f0": r["f0"], "loss_db": round(r["loss"], 2)} for r in miss],
    }


E2 = 82.4069
R20_STEP = 1.122


def baselines(seeds):
    """Exactitud del corte por método: respeta el presupuesto en lo que hay que proteger (1 dB en la nota
    de una línea tonal, 3 dB en el borde útil de una señal percusiva; tolerancia 0.05 dB) y, además, no
    desperdicia más de un paso R20 de limpieza."""
    rows = [r for s in seeds for r in json.load(open(RESULTS / f"baselines_{s}.json"))["cases"]]
    groups = {"todos": rows,
              "tonal": [r for r in rows if r["kind"] == "tonal"],
              "tonal_grave": [r for r in rows if r["kind"] == "tonal" and r["low"] < E2],
              "tonal_medio": [r for r in rows if r["kind"] == "tonal" and r["low"] >= E2],
              "percusion": [r for r in rows if r["kind"] == "perc"]}
    out = {}
    for g, rs in groups.items():
        out[g] = {"n": len(rs)}
        ref_exact = [r["recomendador"]["loss"] <= r["budget"] + 0.05 and r["recomendador"]["ratio"] >= 1 / R20_STEP
                     for r in rs]
        for m in METHODS:
            protect = [r[m]["loss"] <= r["budget"] + 0.05 for r in rs]
            exact = [p and r[m]["ratio"] >= 1 / R20_STEP for p, r in zip(protect, rs)]
            removed = np.array([r[m]["removed"] for r in rs])
            out[g][m] = {"protect": accuracy(protect), "exact": accuracy(exact),
                         "removed_median": float(np.median(removed)),
                         "ratio_median": float(np.median([r[m]["ratio"] for r in rs])),
                         "loss_max": float(max(r[m]["loss"] for r in rs))}
            if m != "recomendador":
                out[g][m]["mcnemar_exact"] = dict(zip(("solo_recomendador", "solo_metodo", "p"),
                                                      mcnemar(ref_exact, exact)))
        out[g]["recomendador"].update(decisions(rs, lambda r: r))
    return out


def decisions(rows, of):
    """Qué decidió el recomendador además del corte: si filtrar y con qué pendiente."""
    active = [bool(of(r).get("active", True)) for r in rows]
    orders = [r_m["order"] for r_m in (of(r)["recomendador"] for r in rows) if r_m["fc"] is not None]
    return {"active": accuracy(active), "orders": {str(n): orders.count(n) for n in sorted(set(orders))}}


def ablation_slope(seeds):
    """Casos cuyo punto protegido es el borde: con el tope de 24 dB/oct y sin él (ablation_slope.py)."""
    paths = [RESULTS / f"ablation_slope_{s}.json" for s in seeds]
    if not all(p.exists() for p in paths):
        return None
    rows = [r for p in paths for r in json.load(open(p))["cases"]]
    out = {"n": len(rows)}
    for name in ("con_tope", "sin_tope"):
        protect = [r[name]["loss"] <= r["budget"] + 0.05 for r in rows]
        exact = [pr and r[name]["ratio"] >= 1 / R20_STEP for pr, r in zip(protect, rows)]
        orders = [r[name]["order"] for r in rows if r[name]["fc"] is not None]
        out[name] = {"protect": accuracy(protect), "exact": accuracy(exact),
                     "removed_median": float(np.median([r[name]["removed"] for r in rows])),
                     "over_24": sum(n > 4 for n in orders)}
    a = [r["con_tope"]["loss"] <= r["budget"] + 0.05 for r in rows]
    b = [r["sin_tope"]["loss"] <= r["budget"] + 0.05 for r in rows]
    out["mcnemar_protect"] = dict(zip(("solo_con_tope", "solo_sin_tope", "p"), mcnemar(a, b)))
    return out


def real_annotated():
    """Grabaciones reales con notas anotadas (real_annotated.py). Además de la exactitud por método,
    compara la nota más grave que encontró el recomendador con la anotada: coincide a menos de 50
    cents, es más grave (protege algo que la anotación no tiene) o más aguda."""
    out = {}
    for path in sorted(RESULTS.glob("real_*.json")):
        data = json.load(open(path))
        items = data["items"]
        out[data["dataset"]] = block = {"n": len(items)}
        for cond in ("original", "contaminado"):
            res = {}
            for m in METHODS:
                protect = [it[cond][m]["loss"] <= it["budget"] + 0.05 for it in items]
                exact = [pr and it[cond][m]["ratio"] >= 1 / R20_STEP for pr, it in zip(protect, items)]
                res[m] = {"protect": accuracy(protect), "exact": accuracy(exact),
                          "ratio_median": float(np.median([it[cond][m]["ratio"] for it in items])),
                          "loss_max": float(max(it[cond][m]["loss"] for it in items))}
                if cond == "contaminado":
                    res[m]["removed_median"] = float(np.median([it[cond][m]["removed"] for it in items]))
            res["recomendador"].update(decisions(items, lambda it: it[cond]))
            cents = [1200 * np.log2(it[cond]["f0_min"] / it["truth"]) if it[cond]["f0_min"] else None for it in items]
            res["f0_vs_annotation"] = {"same": int(sum(c is not None and abs(c) <= 50 for c in cents)),
                                       "lower": int(sum(c is not None and c < -50 for c in cents)),
                                       "higher": int(sum(c is not None and c > 50 for c in cents)),
                                       "none": int(sum(c is None for c in cents))}
            block[cond] = res
        tuning = [it["tuning_cents"]["k1"] for it in items if it.get("tuning_cents")]
        if tuning:
            block["tuning_cents_k1"] = {"min": min(tuning), "max": max(tuning)}
    return out


summary = {"dev_2026": battery(DEV), "design": battery(DESIGN), "test": battery(TEST),
           "baselines_design": baselines(DESIGN), "baselines_test": baselines(TEST), "real": real_annotated(),
           "ablation_slope_design": ablation_slope(DESIGN), "ablation_slope_test": ablation_slope(TEST),
           "fixed_cases": json.load(open(RESULTS / "fixed_cases.json"))}
tracks = {}
for p in sorted(glob.glob(str(RESULTS / "tracks" / "*.json"))):
    d = json.load(open(p))
    v = d["report"]
    tracks[Path(p).stem] = {"dur_s": round(d["dur_s"], 2), "subtype": d["subtype"], "fc": v["cutoff_hz"],
                            "order": v["order"], "active": v.get("active"), "keep": v["keep_hz"], "source": v["source"], "f0": v["f0_min_hz"],
                            "note": v["f0_note"], "voiced": v["voiced_energy"], "edge": v["spectral_edge_hz"],
                            "loss": v["loss_at_keep_db"], "phase": v["phase_at_keep_deg"], "gd_ms": v["group_delay_ms"],
                            "hum": v["hum"], "warnings": v["warnings"]}
summary["tracks"] = tracks
for name in ("floor_calibration", "limits", "cross_track", "session_energy"):
    summary[name] = json.load(open(RESULTS / f"{name}.json"))
(RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

for part in ("dev_2026", "design", "test"):
    b = summary[part]
    print(f"\n== {part}: {b['n_tonal']} tonales, {b['n_perc']} percusivos")
    for key, label in (("", "método completo"), ("_ablation", "sin prueba por trama")):
        a, s = b[f"acc{key}"], b[f"loss{key}"]
        print(f"  {label}: nota más grave {a['k']}/{b['n_tonal']} = {a['pct']:.2f} % "
              f"(Wilson {a['wilson'][0]:.1f}-{a['wilson'][1]:.1f}); pérdida mediana {s['median']:.2f}, "
              f"máx {s['max']:.2f} dB, > 1.05 dB: {s['n_over_1_05']}")
    print(f"  McNemar de la prueba por trama: {b['mcnemar_ablation']}")
    t, pc = b["tonal_rej"], b["perc"]
    print(f"  rechazo tonal: sustituto {t['proxy_median']:.1f}, energía {t['energy_median']:.1f} dB "
          f"(p5 {t['energy_p5']:.1f}); percusión: pérdida mediana {pc['loss_median']:.2f}, máx {pc['loss_max']:.2f}, "
          f"rechazo sustituto {pc['rej_proxy_median']:.1f}, energía {pc['rej_energy_median']:.1f} dB, "
          f"< 6 dB {pc['n_under_6db']}/{b['n_perc']}")
    print(f"  hum: {b['hum']}; fallos: {b['misses']}")
print("\n== exactitud del corte, semillas de prueba (protege / exacto / limpieza mediana)")
for g, block in summary["baselines_test"].items():
    print(f"  {g} (n = {block['n']})")
    for m in METHODS:
        b = block[m]
        print(f"    {m:13} protege {b['protect']['pct']:6.2f} %  exacto {b['exact']['pct']:6.2f} %  "
              f"limpieza {b['removed_median']:5.1f} dB  razón {b['ratio_median']:.2f}")
print(f"\n== pistas: {len(tracks)}")
for k, t in tracks.items():
    print(f"  {k:30} {t['fc']:>5}/{t['order']} keep {t['keep']:>6} {t['source']:8} f0 {t['f0']} {t['note']}")
