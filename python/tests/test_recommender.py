import json

import numpy as np
import pytest
import soundfile as sf

from hpf_recommender import design, recommend
from hpf_recommender.__main__ import main
from signals import SR, bursts, harmonic, rare_low_note_over_hum, rumble

E1, E2 = 41.20, 82.41


def test_silence():
    r = recommend(np.zeros(SR * 2), SR)
    assert r.cutoff_hz is None and r.warnings


def test_nan_is_rejected():
    x = np.zeros(SR)
    x[100] = np.nan
    with pytest.raises(ValueError):
        recommend(x, SR)


def test_bass_with_subsonic_rumble():
    x = harmonic(E1, 6.0) + rumble(6.0, 16.0, 0.01)
    r = recommend(x, SR)
    assert r.source == "f0"
    assert abs(r.keep_hz - E1) < 1.0
    assert r.loss_budget_db == 1.0          # una nota se protege con 1 dB
    assert r.loss_at_keep_db <= 1.0 + 0.02
    assert r.active and r.rejection_met     # el rumble subsónico queda en el objetivo


def test_percussion_edge_clear_of_stationary_noise():
    r = recommend(bursts(8.0) + rumble(8.0, 16.0, 0.001), SR)
    assert r.source == "espectro"
    assert 100.0 < r.keep_hz < 220.0
    # sin altura y con el borde libre de ruido, el corte va en el borde (3 dB); el redondeo R20
    # hacia abajo deja algo menos
    assert r.loss_budget_db == 3.0
    assert 1.0 < r.loss_at_keep_db <= 3.0 + 0.02
    assert not any("reaches the spectral edge" in w for w in r.warnings)
    assert r.active and r.rejection_met       # el rumble de 16 Hz queda en el objetivo


def test_percussion_edge_reached_by_stationary_noise():
    # la falda del rumble de 30 Hz cubre la de las ráfagas hasta ~140 Hz: restar el piso sube el
    # borde medido (124 Hz sin rumble, 135 Hz con él) y ese borde se protege como una nota
    r = recommend(bursts(8.0) + rumble(8.0, 30.0, 0.02), SR)
    assert r.source == "espectro"
    assert 100.0 < r.keep_hz < 220.0     # el rumble de 30 Hz no arrastra el borde hacia abajo
    assert r.cutoff_hz < r.keep_hz
    assert r.loss_budget_db == 1.0
    assert r.loss_at_keep_db <= 1.0 + 0.02
    assert any("reaches the spectral edge" in w for w in r.warnings)


def test_edge_slope_stops_at_24_db_per_octave():
    # el borde medido tiene error; con más pendiente se volvería pérdida. Fijando el orden sí se puede.
    x = bursts(8.0) + rumble(8.0, 60.0, 0.03)
    r = recommend(x, SR)
    assert r.source == "espectro" and r.order <= 4
    assert not r.rejection_met and any("no steeper slope is used" in w for w in r.warnings)
    assert recommend(x, SR, order=8).order == 8


def test_nothing_to_remove_leaves_the_filter_off():
    # sin ruido grave ni offset DC el filtro no quitaría nada medible: se recomienda apagado, con el
    # corte máximo seguro por si igual se quiere filtrar
    r = recommend(harmonic(220.0, 6.0), SR)
    assert not r.active
    assert r.cutoff_hz is not None and r.loss_at_keep_db <= 1.0 + 0.02
    assert any("would remove nothing" in w for w in r.warnings)


def test_dc_offset_turns_the_filter_on_without_the_fixed_filter():
    r = recommend(harmonic(220.0, 6.0) + 0.01, SR, base_hz=None)
    assert r.active and r.base_hz is None and r.base_order == 0
    assert not any("would remove nothing" in w for w in r.warnings)


def test_fixed_20hz_filter_removes_the_dc_offset():
    # el filtro fijo de 20 Hz, siempre puesto, ya quita el offset: no hace falta otro filtro
    r = recommend(harmonic(220.0, 6.0) + 0.01, SR)
    assert r.base_hz == 20.0 and r.base_order == 4
    assert not r.active
    assert any("fixed 20 Hz filter removes it" in w for w in r.warnings)


def test_infrasonic_rumble_is_left_to_the_fixed_filter():
    # sin el filtro fijo, un rumble de 8 Hz bajo E1 enciende el filtro; con él, lo que queda no
    # supera el umbral de contenido útil
    r = recommend(harmonic(E1, 6.0) + rumble(6.0, 8.0, 0.03), SR)
    assert abs(r.keep_hz - E1) < 1.0 and not r.active


def test_infrasonic_rumble_does_not_pull_the_protected_note_down():
    # un rumble fuerte de 8 Hz llevaba la nota protegida a 34 Hz, debajo de E1 (falso hacia abajo)
    r = recommend(harmonic(E1, 6.0) + rumble(6.0, 8.0, 0.1), SR)
    assert r.source == "f0" and abs(r.keep_hz - E1) < 1.0


A_SHARP0 = 29.14


@pytest.mark.parametrize("f0", [30.87, A_SHARP0])
def test_fixed_and_recommended_filters_share_the_budget(f0):
    # el recomendado va detrás del fijo: entre los dos no pasan de 1 dB en la nota, y el corte
    # mínimo (fc_min) no puede romper ese presupuesto
    r = recommend(harmonic(f0, 6.0) + rumble(6.0, 16.0, 0.01), SR)
    assert abs(r.keep_hz - f0) < 1.0 and r.active
    chain = design.attenuation_db(r.keep_hz, 20.0, 4) + design.attenuation_db(r.keep_hz, r.cutoff_hz, r.order)
    assert chain <= 1.0 + 0.02 and r.loss_at_keep_db <= 1.0 + 0.02


def test_gentlest_slope_that_meets_the_target():
    # un rumble débil y lejos de la nota se va con 6 dB/oct
    r = recommend(harmonic(220.0, 6.0) + rumble(6.0, 16.0, 0.01), SR)
    assert r.active and r.rejection_met
    assert r.order == 1 and r.slope_db_oct == 6


def test_steeper_than_24_db_per_octave_when_needed():
    # rumble fuerte poco más de media octava bajo E2: con 24 dB/oct no alcanza, con más pendiente sí
    x = harmonic(E2, 6.0) + rumble(6.0, 45.0, 0.03)
    r = recommend(x, SR)
    assert r.source == "f0" and abs(r.keep_hz - E2) < 1.0
    assert r.order > 4 and r.rejection_met
    assert r.loss_at_keep_db <= 1.0 + 0.02
    assert not recommend(x, SR, max_order=4).rejection_met


def test_hum_series_is_reported():
    t = np.arange(8 * SR) / SR
    hum = sum(0.003 / k * np.sin(2 * np.pi * 60.0 * k * t) for k in (1, 2, 3))
    r = recommend(bursts(8.0) + hum, SR)
    assert r.hum is not None and r.hum["base_hz"] == 60.0
    assert 60.0 in r.hum["harmonics_hz"]


def test_antiphase_stereo_is_not_cancelled():
    s = bursts(6.0) + harmonic(110.0, 6.0)
    mono = recommend(s, SR)
    stereo = recommend(np.column_stack([s, -s]), SR)
    assert stereo.cutoff_hz == mono.cutoff_hz
    assert stereo.keep_hz == pytest.approx(mono.keep_hz, rel=0.02)


def test_short_signal_does_not_crash():
    r = recommend(harmonic(110.0, 0.3), SR)
    assert any("3 s" in w for w in r.warnings)


@pytest.mark.parametrize("subtype", ["PCM_24", "FLOAT"])
def test_cli_export_keeps_format(tmp_path, subtype):
    x = 0.5 * np.column_stack([harmonic(E1, 4.0) + rumble(4.0, 16.0, 0.01), harmonic(E1 * 1.5, 4.0)])
    src = tmp_path / "bajo.wav"
    sf.write(src, x, SR, subtype=subtype)
    main([str(src), "--export", "--json", str(tmp_path / "r.json")])
    out = [p for p in tmp_path.glob("bajo_HPF_*.wav")]
    assert len(out) == 1
    info = sf.info(str(out[0]))
    assert info.subtype == subtype and info.channels == 2 and info.samplerate == SR


def test_cli_exports_only_the_fixed_filter_when_nothing_to_remove(tmp_path):
    src = tmp_path / "la3.wav"
    x = harmonic(220.0, 4.0)
    sf.write(src, x, SR, subtype="FLOAT")
    main([str(src), "--export", "--json", str(tmp_path / "r.json")])
    assert [p.name for p in tmp_path.glob("la3_HPF_*.wav")] == ["la3_HPF_20Hz_ord4.wav"]
    assert json.loads((tmp_path / "r.json").read_text())["active"] is False
    y, _ = sf.read(str(tmp_path / "la3_HPF_20Hz_ord4.wav"))
    assert np.max(np.abs(y - design.apply_hpf(x, SR, 20.0, 4))) < 1e-6


def _line(notes, dur):
    return np.concatenate([harmonic(f, dur) for f in notes])


def test_narrowband_rumble_is_not_a_note():
    # Línea que empieza en E2 sobre un rumble de banda angosta en 45 Hz: YIN lo ve casi
    # periódico y los parciales de la línea le prestan "armónicos" por coincidencia.
    line = _line([E2, 98.0, 110.0, 146.8, 123.5, 98.0] * 2, 0.6)
    x = line + rumble(len(line) / SR, 45.0, 0.02)
    r = recommend(x, SR)
    assert abs(1200 * np.log2(r.keep_hz / E2)) < 50


def test_drone_is_kept():
    # Pedal en E1 todo el tema con notas que se mueven encima: la banda de E1 es
    # estacionaria, pero también la de sus armónicos, así que no se confunde con rumble.
    dur = 7.2
    pedal = harmonic(E1, dur, amp=0.15)
    moving = _line([E2 * 2, 196.0, 220.0, 246.9] * 3, dur / 12)
    x = pedal + moving[:len(pedal)] + rumble(dur, 22.0, 0.005)
    r = recommend(x, SR)
    assert abs(1200 * np.log2(r.keep_hz / E1)) < 50


def test_rare_low_note_over_hum_is_kept():
    # La nota más grave suena el 3 % del tiempo: la media de su banda apenas sube sobre el piso
    # (que además incluye el hum), pero mientras suena, la banda está muy por encima.
    low, x = rare_low_note_over_hum()
    r = recommend(x, SR)
    assert abs(1200 * np.log2(r.keep_hz / low)) < 50
    assert design.attenuation_db(low, r.cutoff_hz, r.order) <= 1.0 + 0.02


def test_short_burst_is_not_silence():
    # 2 ms a -10 dBFS no son silencio digital. Rellenada al final de la trama, la señal quedaría
    # donde la ventana de Hann vale casi cero.
    x = 0.3 * np.sin(2 * np.pi * 1000.0 * np.arange(100) / SR)
    r = recommend(x, SR)
    assert r.cutoff_hz is not None
    assert not any("Digital silence" in w for w in r.warnings)


def test_silence_keeps_dc_and_params():
    # mismo reporte que C++: el offset y los parámetros también en el caso de silencio
    d = recommend(np.zeros(SR * 2), SR).as_dict()
    assert d["dc_offset_dbfs"] == [-240.0]
    assert d["params"]["range_db"] == 30.0


@pytest.mark.parametrize("kw", [dict(fmin=0.0), dict(fmax=20.0), dict(fc_max=10.0), dict(max_order=0),
                                dict(max_order=9), dict(order=9), dict(max_loss_db=-1.0), dict(edge_loss_db=0.0),
                                dict(order=-2)])
def test_invalid_params_raise(kw):
    with pytest.raises(ValueError):
        recommend(harmonic(E1, 1.0), SR, **kw)


def test_cli_continues_after_bad_file(tmp_path, capsys):
    good = tmp_path / "bueno.wav"
    sf.write(good, harmonic(E1, 4.0), SR, subtype="PCM_24")
    bad = tmp_path / "malo.wav"
    bad.write_bytes(b"RIFF\x04\x00\x00\x00WAVE")
    nan = tmp_path / "nan.wav"
    y = harmonic(E1, 4.0)
    y[100] = np.nan
    sf.write(nan, y, SR, subtype="FLOAT")
    status = main([str(bad), str(nan), str(good)])
    out = json.loads(capsys.readouterr().out)
    assert status == 1
    assert out["file"] == "bueno.wav" and out["cutoff_hz"] is not None


def test_low_antiphase_bass_is_kept():
    # Platos en fase y un bajo en contrafase entre L y R, 12 dB abajo: la suma (L+R)/2 no tiene el
    # bajo, pero el espectro sí. Sin la diferencia L-R no hay nota, la regla de la isla tira el bajo
    # y el corte sube a 280 Hz (66 dB de pérdida en E1).
    hats = bursts(6.0, band=(2000.0, 12000.0))
    bass = 10 ** (-12 / 20) * harmonic(E1, 6.0)
    r = recommend(np.column_stack([hats + bass, hats - bass]), SR)
    assert abs(1200 * np.log2(r.keep_hz / E1)) < 50
    assert design.attenuation_db(E1, r.cutoff_hz, r.order) <= 1.0 + 0.02
