// Nota más grave con YIN (de Cheveigné y Kawahara, 2002, JASA 111(4)) y verificación espectral.
#include "pitch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "analysis.h"
#include "dsp.h"

namespace hpf::pitch {

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();
}

YinResult yin(const std::vector<double>& x, double sr, double fmin, double fmax, double hop_s, double threshold) {
    YinResult r;
    const long tau_min = std::max(2L, static_cast<long>(sr / fmax));
    const long tau_max = static_cast<long>(std::ceil(sr / fmin));
    const long w = tau_max;   // ventana de integración = periodo más largo buscado
    const long flen = w + tau_max;
    const long hop = std::max(1L, static_cast<long>(std::nearbyint(hop_s * sr)));
    const long n = static_cast<long>(x.size());
    if (n < flen) return r;

    const long n_frames = (n - flen) / hop + 1;
    const std::size_t nfft = dsp::next_pow2(double(flen + w));
    dsp::RealFft fft(nfft);
    const std::size_t nb = nfft / 2 + 1;
    std::vector<dsp::cplx> a(nb), b(nb);
    std::vector<double> rr(nfft), cs(flen + 1), d(tau_max + 1), cm(tau_max + 1);

    r.f0.assign(n_frames, kNaN);
    r.power.assign(n_frames, 0.0);
    r.aper.assign(n_frames, kNaN);
    r.aper2.assign(n_frames, kNaN);
    r.centers.resize(n_frames);

    for (long s = 0; s < n_frames; ++s) {
        const double* fr = x.data() + s * hop;
        r.centers[s] = s * hop + flen / 2;
        cs[0] = 0.0;
        for (long i = 0; i < flen; ++i) cs[i + 1] = cs[i] + fr[i] * fr[i];
        r.power[s] = cs[w] / double(w);

        // d(tau) = sum_j (x_j - x_{j+tau})^2 desarrollado; la correlación cruzada por FFT
        fft.forward(fr, flen, a.data());
        fft.forward(fr, w, b.data());
        for (std::size_t k = 0; k < nb; ++k)   // a·conj(b) a mano (ver dsp.cpp)
            a[k] = dsp::cplx(a[k].real() * b[k].real() + a[k].imag() * b[k].imag(),
                             a[k].imag() * b[k].real() - a[k].real() * b[k].imag());
        fft.inverse(a.data(), rr.data());
        for (long t = 0; t <= tau_max; ++t) d[t] = cs[w] + cs[t + w] - cs[t] - 2.0 * rr[t];
        d[0] = 0.0;
        cm[0] = 1.0;
        double run = 0.0;
        for (long t = 1; t <= tau_max; ++t) {
            run += d[t];
            cm[t] = d[t] * double(t) / std::max(run, 1e-12);
        }

        long t = -1;
        for (long q = tau_min; q < tau_max; ++q)
            if (cm[q] < threshold) { t = q; break; }
        if (t < 0) continue;
        while (t + 1 < tau_max && cm[t + 1] < cm[t]) ++t;   // del primer cruce baja al mínimo local
        if (!(t < tau_max - 1)) continue;                    // pegado al borde de búsqueda

        const double ca = cm[t - 1], cb = cm[t], cc = cm[std::min(t + 1, tau_max)];
        const double den = ca - 2.0 * cb + cc;
        double shift = std::abs(den) > 1e-12 ? 0.5 * (ca - cc) / den : 0.0;
        shift = std::clamp(shift, -1.0, 1.0);
        r.f0[s] = sr / (double(t) + shift);
        r.aper[s] = cb;
        // en el periodo doble: mínimo de t2-1, t2, t2+1 (2t no cae justo en un entero)
        if (2 * t + 1 <= tau_max) {
            const long t2 = std::min(2 * t, tau_max - 1);
            r.aper2[s] = std::min({cm[t2 - 1], cm[t2], cm[t2 + 1]});
        }
    }
    return r;
}

std::vector<double> verify_octave(const std::vector<double>& x, double sr, const std::vector<double>& f0,
                                  const std::vector<long>& centers, double fmin, const std::vector<double>* aper,
                                  const std::vector<double>* aper2, int n_harm, double prom_db, double drop_db,
                                  std::size_t nfft) {
    // Corrige errores de octava de YIN con el espectro de la misma trama. Hacia arriba: el
    // múltiplo k·f0 más bajo que es una nota creíble (pico propio, a menos de drop_db del
    // parcial más fuerte, y pico también en 2 o 3 veces su frecuencia). Hacia abajo, f0/2,
    // solo si además YIN lo respalda: d'(2t) < d'(t).
    std::vector<double> out(f0.size(), kNaN);
    const std::size_t half = nfft / 2;
    const auto win = dsp::hann_periodic(nfft);
    const double res = sr / double(nfft);   // resolución real; el zero-padding x2 solo interpola
    const double bin_hz = res / 2.0;
    const double f_lim = 0.45 * sr;
    dsp::RealFft fft(2 * nfft);
    std::vector<double> seg(nfft), sp(nfft + 1);
    std::vector<dsp::cplx> spec(nfft + 1);
    const long nx = static_cast<long>(x.size());

    auto slice_end = [&](long hi) { return static_cast<std::size_t>(std::min<long>(hi, long(sp.size()))); };
    struct Peak { double level; bool prominent; };
    auto peak = [&](double f) -> Peak {
        const double tol = std::max(0.03 * f, 0.5 * res);
        const long lo = static_cast<long>((f - tol) / bin_hz);
        const long hi = static_cast<long>(std::ceil((f + tol) / bin_hz)) + 1;
        const std::size_t e = slice_end(hi);
        std::size_t m = lo;
        for (std::size_t q = lo; q < e; ++q)
            if (sp[q] > sp[m]) m = q;
        const long flo = static_cast<long>(f / std::sqrt(2.0) / bin_hz);
        const long fhi = static_cast<long>(std::ceil(f * std::sqrt(2.0) / bin_hz)) + 1;
        const double floor = dsp::median_range(sp.data() + flo, sp.data() + slice_end(fhi));
        const std::size_t rel = m - lo, len = e - lo;
        // máximo en el borde de la ventana = ladera de otro pico
        return {sp[m], rel > 0 && rel + 1 < len && sp[m] - floor >= prom_db};
    };
    auto credible = [&](double f, double strongest) {
        const Peak p = peak(f);
        if (!p.prominent || p.level < strongest - drop_db) return false;
        for (int h : {2, 3})
            if (h * f < f_lim && peak(h * f).prominent) return true;
        return false;
    };

    for (std::size_t j = 0; j < f0.size(); ++j) {
        if (!std::isfinite(f0[j])) continue;
        const long c = centers[j];
        for (std::size_t i = 0; i < nfft; ++i) {
            const long k = c + long(i) - long(half);   // x con half ceros a cada lado
            seg[i] = (k >= 0 && k < nx ? x[k] : 0.0) * win[i];
        }
        fft.forward(seg.data(), nfft, spec.data());
        for (std::size_t k = 0; k <= nfft; ++k) sp[k] = 20.0 * std::log10(std::sqrt(std::norm(spec[k])) + 1e-12);

        std::vector<double> fk;
        for (int k = 1; k <= n_harm; ++k)
            if (f0[j] * k < f_lim) fk.push_back(f0[j] * k);
        double strongest = -1e300;
        for (double f : fk) strongest = std::max(strongest, peak(f).level);
        const double h = 0.5 * f0[j];
        if (h >= fmin && aper2 && std::isfinite((*aper2)[j]) && (*aper2)[j] < (*aper)[j] && credible(h, strongest)) {
            out[j] = h;
            continue;
        }
        for (double f : fk)
            if (credible(f, strongest)) { out[j] = f; break; }
    }
    return out;
}

namespace {

int longest_run(std::vector<long> frames, long max_gap = 2) {
    std::sort(frames.begin(), frames.end());
    int best = 0, cur = 0;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        cur = (i > 0 && frames[i] - frames[i - 1] <= max_gap) ? cur + 1 : 1;
        best = std::max(best, cur);
    }
    return best;
}

}  // namespace

std::optional<double> lowest_note(const std::vector<double>& f0, int min_frames, int min_run, double width_cents) {
    // f0 más grave con soporte: min_frames tramas dentro de un semitono y un tramo de
    // min_run tramas casi seguidas. Sin proporción mínima: la nota más grave de una línea
    // de bajo puede sonar 1 % del tiempo y sigue siendo la que hay que proteger.
    std::vector<std::pair<double, long>> v;   // (cents, trama)
    for (std::size_t i = 0; i < f0.size(); ++i)
        if (std::isfinite(f0[i])) v.emplace_back(1200.0 * std::log2(f0[i] / 440.0), long(i));
    if (long(v.size()) < min_frames) return std::nullopt;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    const std::size_t n = v.size();
    std::size_t end = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double lim = v[i].first + width_cents;
        end = std::max(end, i);
        while (end < n && v[end].first <= lim) ++end;
        if (long(end - i) < min_frames) continue;
        std::vector<long> frames;
        std::vector<double> vals;
        for (std::size_t q = i; q < end; ++q) {
            frames.push_back(v[q].second);
            vals.push_back(f0[v[q].second]);
        }
        if (longest_run(frames) >= min_run) return dsp::median(vals);
    }
    return std::nullopt;
}

namespace {

// Señales para f0. En estéreo, la suma (L+R)/2 y la diferencia (L-R)/2: lo que está en contrafase
// desaparece de la suma, pero el espectro (que suma los canales en potencia) sí lo ve, y sin nota la
// regla de la isla puede tirarlo. Con más de dos canales, la media o, si se cancela, el canal más fuerte.
std::vector<std::vector<double>> pitch_signals(const Channels& x) {
    const std::size_t n = x.empty() ? 0 : x[0].size();
    if (x.size() == 1) return {x[0]};
    if (x.size() == 2) {
        std::vector<double> m(n), d(n);
        for (std::size_t i = 0; i < n; ++i) {
            m[i] = 0.5 * (x[0][i] + x[1][i]);
            d[i] = 0.5 * (x[0][i] - x[1][i]);
        }
        return {m, d};
    }
    std::vector<double> mono(n, 0.0), per(x.size(), 0.0);
    for (std::size_t c = 0; c < x.size(); ++c) {
        for (std::size_t i = 0; i < n; ++i) {
            mono[i] += x[c][i];
            per[c] += x[c][i] * x[c][i];
        }
        per[c] /= double(std::max<std::size_t>(n, 1));
    }
    double mid = 0.0;
    for (auto& v : mono) { v /= double(x.size()); mid += v * v; }
    mid /= double(std::max<std::size_t>(n, 1));
    const double mean_per = std::accumulate(per.begin(), per.end(), 0.0) / double(per.size());
    if (mid < 0.25 * mean_per) return {x[std::max_element(per.begin(), per.end()) - per.begin()]};
    return {mono};
}

// Verificación de octava y exclusiones (hum, bandas estacionarias) de una pista de f0.
std::vector<double> verified_f0(const std::vector<double>& y, const YinResult& yr, double fmin, double fmax,
                                const std::vector<double>& exclude_hz, const Steady* steady) {
    auto f0 = verify_octave(y, kAnalysisSr, yr.f0, yr.centers, fmin, &yr.aper, &yr.aper2);
    for (auto& v : f0)
        if (v > fmax) v = kNaN;
    // Un zumbido estacionario es periódico y YIN lo toma por nota: fuera las tramas a menos
    // de 25 cents de un armónico de red detectado.
    for (double h : exclude_hz)
        for (auto& v : f0)
            if (std::isfinite(v) && std::abs(1200.0 * std::log2(v / h)) < 25.0) v = kNaN;
    // Rumble de banda angosta: casi periódico, con f0 en cualquier parte. Una nota real entra y
    // sale: mientras suena, su banda queda muy por encima del piso, aunque la media de largo
    // plazo casi no se mueva. Se descarta la trama si en ese momento su banda está a menos de
    // kFrameSnrDb del piso (un rumble gaussiano con ~2 grados de libertad pasa 10 dB sobre su media
    // con probabilidad e^-10). Una nota sostenida (pedal) también es estacionaria, pero arrastra sus
    // armónicos estacionarios.
    if (steady) {
        const auto& cen = *steady->centers;
        auto band = [&](double f) {
            auto it = std::lower_bound(cen.begin(), cen.end(), f * std::pow(2.0, -1.0 / 48.0));
            return std::min<std::size_t>(it - cen.begin(), cen.size() - 1);
        };
        const auto& fdb = *steady->frame_db;
        for (std::size_t i = 0; i < f0.size(); ++i) {
            double& v = f0[i];
            if (!std::isfinite(v)) continue;
            const std::size_t b = band(v);
            const bool drone = steady->mask[band(2 * v)] || steady->mask[band(3 * v)];
            if (!steady->mask[b] || drone) continue;
            const double t = double(yr.centers[i]) / kAnalysisSr;
            const double kf = std::nearbyint((t - steady->t0) / steady->hop);
            const std::size_t k = static_cast<std::size_t>(std::clamp(kf, 0.0, double(fdb.size() - 1)));
            if (fdb[k][b] - (*steady->floor_db)[b] < kFrameSnrDb) v = kNaN;
        }
    }
    return f0;
}

}  // namespace

PitchResult estimate_min_f0(const Channels& x, double sr, double fmin, double fmax, double gate_db,
                            const std::vector<double>& exclude_hz, const Steady* steady) {
    const long g = std::gcd(static_cast<long>(sr), long(kAnalysisSr));
    struct Track {
        std::vector<double> y;
        YinResult r;
        std::vector<double> p_db;
    };
    std::vector<Track> tracks;
    for (const auto& s : pitch_signals(x)) {
        Track t;
        t.y = dsp::resample_poly(s, kAnalysisSr / g, static_cast<long>(sr) / g);
        t.r = yin(t.y, kAnalysisSr, fmin, fmax);
        tracks.push_back(std::move(t));
    }
    PitchResult out;
    if (tracks.empty() || tracks[0].r.f0.empty()) return out;

    // Solo tramas a menos de gate_db del nivel alto del tema (el mismo umbral para suma y
    // diferencia): lo que queda debajo es cola de reverb, ruido o sangrado lejano.
    double p95 = -1e300;
    bool any = false;
    for (auto& t : tracks) {
        std::vector<double> valid_db;
        t.p_db.resize(t.r.power.size());
        for (std::size_t i = 0; i < t.p_db.size(); ++i) {
            t.p_db[i] = analysis::to_db(t.r.power[i]);
            if (t.p_db[i] > analysis::kSilenceDb) valid_db.push_back(t.p_db[i]);
        }
        if (!valid_db.empty()) {
            p95 = std::max(p95, dsp::percentile(valid_db, 95.0));
            any = true;
        }
    }
    if (!any) return out;
    const double gate = p95 - gate_db;

    // En energía y no en tiempo: en una batería las tramas con nota son colas o sangrado. La misma
    // trama en la suma y en la diferencia cuenta una vez (queda el f0 más bajo de las dos).
    std::vector<double> merged(tracks[0].r.f0.size(), kNaN);
    double e_voiced = 0.0, e_active = 0.0;
    for (auto& t : tracks) {
        std::vector<bool> active(t.p_db.size());
        for (std::size_t i = 0; i < t.p_db.size(); ++i) {
            active[i] = t.p_db[i] > analysis::kSilenceDb && t.p_db[i] >= gate;
            if (!active[i]) t.r.f0[i] = kNaN;
        }
        const auto f0 = verified_f0(t.y, t.r, fmin, fmax, exclude_hz, steady);
        for (std::size_t i = 0; i < f0.size(); ++i) {
            if (active[i]) e_active += t.r.power[i];
            if (!std::isfinite(f0[i])) continue;
            e_voiced += t.r.power[i];
            if (!std::isfinite(merged[i]) || f0[i] < merged[i]) merged[i] = f0[i];
        }
    }
    for (double v : merged) out.n_voiced += std::isfinite(v) ? 1 : 0;
    out.voiced_energy = e_active > 0 ? e_voiced / e_active : 0.0;
    out.f0_min = lowest_note(merged);
    return out;
}
}  // namespace hpf::pitch
