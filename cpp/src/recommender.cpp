#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "analysis.h"
#include "design.h"
#include "dsp.h"
#include "hpf/hpf.h"
#include "pitch.h"

namespace hpf {

namespace {

double round_to(double v, int digits) {
    const double s = std::pow(10.0, digits);
    return std::nearbyint(v * s) / s;
}

std::string fmt(const char* f, double v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, f, v);
    return buf;
}

}  // namespace

std::string note_name(double f) {
    static const char* names[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    const int n = static_cast<int>(std::nearbyint(12.0 * std::log2(f / 440.0))) + 57;   // A4 = 57 sobre C0
    const int pc = ((n % 12) + 12) % 12;
    const int octave = (n - pc) / 12;
    return std::string(names[pc]) + std::to_string(octave);
}

namespace {

// Lo que el resto del código da por cierto. Fuera de esto hay conversiones de infinito a entero
// (fmin, fmax: comportamiento indefinido), std::clamp con límites al revés (también indefinido)
// o ningún orden que probar.
void check_params(double sr, const Params& p) {
    for (double v : {sr, p.max_loss_db, p.edge_loss_db, p.range_db, p.fmin, p.fmax, p.fc_min, p.fc_max, p.f_floor})
        if (!std::isfinite(v)) throw std::invalid_argument("sample rate or parameters not finite");
    if (!(sr > 0)) throw std::invalid_argument("empty audio or invalid sample rate");
    if (!(p.max_loss_db > 0 && p.edge_loss_db > 0 && p.range_db > 0))
        throw std::invalid_argument("max_loss_db, edge_loss_db and range_db must be positive");
    if (!(p.fmin > 0 && p.fmin < p.fmax)) throw std::invalid_argument("0 < fmin < fmax is required");
    if (!(p.fc_min > 0 && p.fc_min <= p.fc_max)) throw std::invalid_argument("0 < fc_min <= fc_max is required");
    if (p.order < 0 || p.order > max_filter_order)
        throw std::invalid_argument("order goes from 1 to 8 (0: the lowest that meets the target)");
    if (p.order == 0 && (p.max_order < 1 || p.max_order > max_filter_order))
        throw std::invalid_argument("max_order goes from 1 to 8");
    if (!std::isfinite(p.base_hz) || p.base_hz < 0 || (p.base_hz > 0 && !(p.base_hz < 0.5 * sr)))
        throw std::invalid_argument("base_hz goes from 0 (no fixed filter) to sr/2");
}

}  // namespace

Recommendation recommend(const Channels& x, double sr, const Params& p) {
    if (x.empty()) throw std::invalid_argument("empty audio or invalid sample rate");
    check_params(sr, p);
    const std::size_t n = x[0].size();
    for (const auto& ch : x) {
        if (ch.size() != n) throw std::invalid_argument("channels of different length");
        for (double v : ch)
            if (!std::isfinite(v)) throw std::invalid_argument("the signal contains NaN or Inf");
    }

    Recommendation r;
    r.params = p;
    const int base_order = p.base_hz > 0 ? base_filter_order : 0;
    if (p.base_hz > 0) r.base_hz = p.base_hz;
    r.base_order = base_order;
    double max_dc = -1e300;
    for (const auto& ch : x) {
        double m = 0.0;
        for (double v : ch) m += v;
        m /= double(std::max<std::size_t>(n, 1));
        const double db = 20.0 * std::log10(std::abs(m) + 1e-12);
        r.dc_offset_dbfs.push_back(round_to(db, 1));
        max_dc = std::max(max_dc, db);
    }
    const bool has_dc = max_dc > -60.0;
    if (has_dc)
        r.warnings.push_back("DC offset of " + fmt("%.0f", max_dc) + " dBFS; " +
                             (p.base_hz > 0 ? "the fixed " + fmt("%g", p.base_hz) + " Hz filter removes it."
                                            : std::string("the HPF removes it.")));
    if (double(n) < 3 * sr) r.warnings.push_back("Signal shorter than 3 s: the estimate is unreliable.");

    // El análisis se hace sobre la señal ya filtrada por el fijo: lo de debajo no decide nada.
    Channels xb;
    if (p.base_hz > 0) {
        xb = x;
        apply_hpf(xb, sr, p.base_hz, base_order);
    }
    const Channels& xa = p.base_hz > 0 ? xb : x;
    const auto spec = analysis::long_term_spectrum(xa, sr);
    if (!spec) {
        r.warnings.push_back("Digital silence: nothing to analyse.");
        return r;
    }
    if (spec->stationary)
        r.warnings.push_back("Stationary material (no dynamics): background noise cannot be separated from the signal.");

    const double f_top = std::min(16000.0, 0.45 * sr);
    const double edge = analysis::spectral_edge(*spec, p.f_floor, f_top, p.range_db);
    const std::size_t ref = spec->peak_index(p.f_floor, f_top);
    auto hum = analysis::detect_hum(*spec);

    // Bandas donde el piso estacionario es más de la mitad de la media: ahí no hay notas
    // que entran y salen, hay rumble, hum o un tono de equipo.
    pitch::Steady steady{&spec->centers, {}, &spec->frame_db, &spec->floor_db, spec->frame_t0, spec->frame_hop};
    for (std::size_t b = 0; b < spec->centers.size(); ++b)
        steady.mask.push_back(spec->floor_db[b] > spec->mean_db[b] - 3.0);
    const auto pr = pitch::estimate_min_f0(xa, sr, p.fmin, p.fmax, 35.0,
                                           hum ? hum->harmonics_hz : std::vector<double>{},
                                           spec->stationary ? nullptr : &steady);

    // Con una fuente afinada manda la nota más grave: debajo de su fundamental el
    // instrumento no produce parciales. Si las tramas con nota pesan poco en la energía
    // (batería, ambiente), el f0 solo puede bajar el punto protegido.
    double keep;
    if (pr.f0_min && pr.voiced_energy >= p.tonal_energy) {
        keep = *pr.f0_min;
        r.source = "f0";
    } else if (pr.f0_min && *pr.f0_min < edge) {
        keep = *pr.f0_min;
        r.source = "mixto";
    } else {
        keep = edge;
        r.source = "espectro";
    }
    // Una nota pertenece a la fuente: se protege con max_loss_db. El borde del espectro está, por
    // definición, range_db bajo el máximo, y ahí basta edge_loss_db. Salvo que el ruido estacionario
    // llegue al borde (una banda dominada por el piso a menos de un ancho de banda de análisis, 1/3
    // de octava, debajo de él): al restar el piso se resta también la falda de la fuente, el borde
    // medido es solo una cota superior y se vuelve a max_loss_db.
    double budget = p.max_loss_db;
    if (r.source == "espectro") {
        bool reached = false;
        if (!spec->stationary)
            for (std::size_t b = 0; b < spec->centers.size(); ++b)
                reached = reached || (spec->centers[b] >= edge * std::pow(2.0, -1.0 / 3.0) &&
                                      spec->centers[b] < edge && steady.mask[b]);
        if (reached)
            r.warnings.push_back("Stationary noise reaches the spectral edge (" + fmt("%.0f", edge) +
                                 " Hz): that edge is an upper bound and is protected with " + fmt("%g", p.max_loss_db) +
                                 " dB, not " + fmt("%g", p.edge_loss_db) + " dB.");
        else
            budget = p.edge_loss_db;
    }

    // Ruido grave: bandas enteras por debajo de keep/sqrt(2) que superan el umbral de
    // contenido útil; tras el filtro deben quedar en ref - range_db. La media octava bajo
    // keep es banda de transición: ahí cae la fuga de la propia nota protegida.
    // El recomendado va detrás del fijo: entre los dos no pasan del presupuesto en keep.
    double rec_budget = p.base_hz > 0 ? budget - design::attenuation_db(keep, p.base_hz, base_order) : budget;
    rec_budget = std::max(rec_budget, 1e-6);
    std::vector<double> junk_hz, need;
    for (std::size_t b = 0; b < spec->centers.size(); ++b) {
        if (!(spec->centers[b] * std::pow(2.0, 1.0 / 6.0) < keep / std::sqrt(2.0))) continue;
        const double nd = spec->mean_db[b] - (spec->mean_db[ref] - p.range_db);
        if (nd > 0) {
            junk_hz.push_back(spec->centers[b]);
            need.push_back(nd);
        }
    }
    // Una nota se mide con precisión y admite cualquier pendiente. El borde de una fuente sin altura
    // es una falda gradual que se mide con error (sube cuando el ruido estacionario la cubre), y con
    // más pendiente el mismo error se vuelve más pérdida: ahí la pendiente llega a 24 dB/oct.
    const int top = r.source == "espectro" ? std::min(p.max_order, edge_max_order) : p.max_order;
    std::vector<int> orders;
    if (p.order > 0) orders.push_back(p.order);
    else for (int o = 1; o <= top; ++o) orders.push_back(o);
    // Con el fijo delante, un orden cuyo corte máximo cae bajo fc_min no cabe en el presupuesto
    // (subir el corte a fc_min lo rompería) y lo subsónico ya lo quita el fijo: se descarta.
    if (p.base_hz > 0 && p.order == 0) {
        std::vector<int> fits;
        for (int o : orders)
            if (design::max_cutoff(keep, o, rec_budget) >= p.fc_min) fits.push_back(o);
        if (!fits.empty()) orders = fits;
    }
    const auto c = design::choose(keep, junk_hz, need, rec_budget, orders, p.fc_min, p.fc_max, p.snap);
    if (!c.met) {
        const std::string fix = r.source == "espectro" && p.order == 0 && top < p.max_order
                                    ? "on the edge of an unpitched source no steeper slope is used: treat it separately."
                                    : "a steeper slope or separate treatment is needed.";
        r.warnings.push_back("At " + std::to_string(6 * c.order) + " dB/oct the low-frequency noise stays " +
                             fmt("%.0f", c.short_db) + " dB above the target; " + fix);
    }
    if (design::max_cutoff(keep, c.order, rec_budget) > p.fc_max)
        r.warnings.push_back("Corner limited to fc_max = " + fmt("%g", p.fc_max) + " Hz.");
    if (design::max_cutoff(keep, c.order, rec_budget) < p.fc_min)
        r.warnings.push_back("Useful content reaches down to " + fmt("%.0f", keep) +
                             " Hz: only a subsonic filter fits, at fc_min = " + fmt("%g", p.fc_min) + " Hz.");
    // Filtrar solo si hay algo medible que quitar: bandas bajo la transición sobre el umbral de
    // contenido útil, u offset DC si no hay filtro fijo que lo quite. Pestana y Reiss (2014) no
    // encontraron justificación subjetiva para el pasa-altos en pistas sin graves.
    r.active = !junk_hz.empty() || (has_dc && !(p.base_hz > 0));
    if (!r.active) {
        const std::string quitar = p.base_hz > 0 ? "what is left after the fixed " + fmt("%g", p.base_hz) + " Hz filter"
                                                 : std::string("no DC offset");
        r.warnings.push_back("Below " + fmt("%.0f", keep / std::sqrt(2.0)) +
                             " Hz nothing exceeds the usefulness threshold (" + quitar + "): the filter would "
                             "remove nothing measurable and leaving it off is recommended (the corner given is the "
                             "highest that respects the allowed loss).");
    }

    const auto resp = design::response_at(keep, c.fc, c.order, sr, p.base_hz, base_order);
    if (hum) {
        Hum h{hum->base_hz, hum->harmonics_hz, round_to(hum->level_db - spec->mean_db[ref], 1)};
        std::vector<double> above;
        for (double f : h.harmonics_hz)
            if (f > c.fc) above.push_back(f);
        if (!above.empty() && h.level_db_re_peak > -60.0) {
            std::string list;
            for (std::size_t i = 0; i < above.size(); ++i) list += (i ? ", " : "") + fmt("%g", above[i]);
            r.warnings.push_back("Stationary tones at multiples of " + fmt("%g", h.base_hz) +
                                 " Hz above the corner (" + list +
                                 " Hz), typical of mains hum: an HPF cannot remove them without cutting the "
                                 "instrument; use a notch or de-hum.");
        }
        r.hum = h;
    }

    r.cutoff_hz = c.fc;
    r.order = c.order;
    r.slope_db_oct = 6 * c.order;
    r.keep_hz = round_to(keep, 1);
    r.loss_at_keep_db = round_to(resp.loss_db, 2);
    r.loss_budget_db = budget;
    r.phase_at_keep_deg = round_to(resp.phase_deg, 1);
    r.group_delay_ms = round_to(resp.group_delay_ms, 2);
    r.rejection_met = c.met;
    r.rejection_short_db = round_to(c.short_db, 1);
    if (pr.f0_min) {
        r.f0_min_hz = round_to(*pr.f0_min, 1);
        r.f0_note = note_name(*pr.f0_min);
    }
    r.voiced_energy = round_to(pr.voiced_energy, 3);
    r.spectral_edge_hz = round_to(edge, 1);
    r.peak_band_hz = round_to(spec->centers[ref], 1);
    return r;
}

void apply_hpf(Channels& x, double sr, double fc, int order) {
    const auto s = design::sos(fc, order, sr);
    for (auto& ch : x) dsp::sosfilt(s, ch.data(), ch.size());
}

void apply_recommendation(Channels& x, double sr, const Recommendation& r) {
    if (r.base_hz) apply_hpf(x, sr, *r.base_hz, r.base_order);
    if (r.active && r.cutoff_hz) apply_hpf(x, sr, *r.cutoff_hz, r.order);
}

namespace {

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char ch : s) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += char(ch); }
        else if (ch == '\n') o += "\\n";
        else if (ch < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
        else o += char(ch);
    }
    return o + "\"";
}

std::string json_num(double v) {
    if (!std::isfinite(v)) return "null";
    // como repr() de Python: los dígitos más cortos que vuelven al mismo double, en notación fija
    // si el exponente decimal está entre -4 y 15 (240.0, 1000.0, 0.0001) y científica fuera de
    // eso (1e-05). Con %g, -240.0 saldría -2.4e+02 y 1000.0, 1e+03.
    char sci[40];
    int prec = 1;
    for (; prec < 17; ++prec) {
        std::snprintf(sci, sizeof sci, "%.*e", prec - 1, v);
        if (std::strtod(sci, nullptr) == v) break;
    }
    std::snprintf(sci, sizeof sci, "%.*e", prec - 1, v);
    const int e = std::atoi(std::strchr(sci, 'e') + 1);
    if (e < -4 || e >= 16) return sci;
    char fix[64];
    std::snprintf(fix, sizeof fix, "%.*f", std::max(prec - 1 - e, 0), v);
    std::string s = fix;
    if (s.find('.') == std::string::npos) s += ".0";
    return s;
}

template <class T>
std::string opt_num(const std::optional<T>& v) { return v ? json_num(*v) : "null"; }

std::string num_list(const std::vector<double>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + json_num(v[i]);
    return s + "]";
}

}  // namespace

std::string json_string(const std::string& s) { return json_str(s); }

std::string to_json(const Recommendation& r, int indent) {
    const std::string pad(indent, ' ');
    std::ostringstream o;
    o << "{\n";
    auto field = [&](const char* k, const std::string& v, bool last = false) {
        o << pad << json_str(k) << ": " << v << (last ? "\n" : ",\n");
    };
    field("cutoff_hz", opt_num(r.cutoff_hz));
    field("order", std::to_string(r.order));
    field("slope_db_oct", std::to_string(r.slope_db_oct));
    field("active", r.active ? "true" : "false");
    field("keep_hz", opt_num(r.keep_hz));
    field("source", json_str(r.source));
    field("loss_at_keep_db", opt_num(r.loss_at_keep_db));
    field("loss_budget_db", opt_num(r.loss_budget_db));
    field("base_hz", opt_num(r.base_hz));
    field("base_order", std::to_string(r.base_order));
    field("phase_at_keep_deg", opt_num(r.phase_at_keep_deg));
    field("group_delay_ms", opt_num(r.group_delay_ms));
    field("rejection_met", r.rejection_met ? "true" : "false");
    field("rejection_short_db", json_num(r.rejection_short_db));
    field("f0_min_hz", opt_num(r.f0_min_hz));
    field("f0_note", r.f0_note ? json_str(*r.f0_note) : "null");
    field("voiced_energy", json_num(r.voiced_energy));
    field("spectral_edge_hz", opt_num(r.spectral_edge_hz));
    field("peak_band_hz", opt_num(r.peak_band_hz));
    if (r.hum)
        field("hum", "{\"base_hz\": " + json_num(r.hum->base_hz) + ", \"harmonics_hz\": " +
                         num_list(r.hum->harmonics_hz) + ", \"level_db_re_peak\": " +
                         json_num(r.hum->level_db_re_peak) + "}");
    else
        field("hum", "null");
    field("dc_offset_dbfs", num_list(r.dc_offset_dbfs));
    std::string w = "[";
    for (std::size_t i = 0; i < r.warnings.size(); ++i) w += (i ? ", " : "") + json_str(r.warnings[i]);
    field("warnings", w + "]");
    const Params& p = r.params;
    field("params", "{\"max_loss_db\": " + json_num(p.max_loss_db) + ", \"edge_loss_db\": " +
                        json_num(p.edge_loss_db) + ", \"range_db\": " + json_num(p.range_db) +
                        ", \"fmin\": " + json_num(p.fmin) + ", \"fmax\": " + json_num(p.fmax) +
                        ", \"max_order\": " + std::to_string(p.max_order) + ", \"fc_max\": " + json_num(p.fc_max) +
                        ", \"base_hz\": " + (p.base_hz > 0 ? json_num(p.base_hz) : std::string("null")) + "}",
          true);
    o << "}";
    return o.str();
}

}  // namespace hpf
