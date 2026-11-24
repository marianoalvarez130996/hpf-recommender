// Espectro de largo plazo en bandas de 1/3 de octava, piso estacionario y zumbido de red.
#include "analysis.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "dsp.h"

namespace hpf::analysis {

double to_db(double p) { return 10.0 * std::log10(std::max(p, 1e-20)); }

std::vector<double> band_centers(double f_lo, double f_hi, int per_octave) {
    const double k0 = std::ceil(per_octave * std::log2(f_lo / 1000.0));
    const double k1 = std::floor(per_octave * std::log2(f_hi / 1000.0));
    std::vector<double> c;
    for (double k = k0; k <= k1; k += 1.0) c.push_back(1000.0 * std::pow(2.0, k / per_octave));
    return c;
}

double band_dof(std::size_t i_lo, double w_lo, std::size_t i_hi, double w_hi) {
    // nu = 2 (sum a)^2 / sum_ij a_i a_j |rho_ij|^2; con Hann periódica la potencia de bins
    // vecinos correlaciona 4/9 y a dos bins 1/36. El ancho entre el ENBW (1.5 bins) no es la
    // distancia de decorrelación.
    std::vector<double> a(i_hi - i_lo + 1, 1.0);
    if (i_hi == i_lo) a[0] = w_hi - w_lo;
    else { a.front() = 1.0 - w_lo; a.back() = w_hi; }
    double sum = 0.0, aa = 0.0, a1 = 0.0, a2 = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        sum += a[k];
        aa += a[k] * a[k];
        if (k + 1 < a.size()) a1 += a[k] * a[k + 1];
        if (k + 2 < a.size()) a2 += a[k] * a[k + 2];
    }
    return 2.0 * sum * sum / (aa + 2.0 * (4.0 / 9.0) * a1 + 2.0 * (1.0 / 36.0) * a2);
}

std::size_t Spectrum::peak_index(double f_lo, double f_hi) const {
    std::size_t best = centers.size();
    for (std::size_t i = 0; i < centers.size(); ++i) {
        if (centers[i] < f_lo || centers[i] > f_hi) continue;
        if (best == centers.size() || excess_db[i] > excess_db[best]) best = i;
    }
    if (best == centers.size()) throw std::runtime_error("sin bandas en el rango pedido");
    return best;
}

namespace {

// Potencia por trama y por banda. Los canales se suman en potencia: un contenido en
// contrafase entre L y R no desaparece, como sí pasaría con un downmix (L+R)/2.
struct FramePowers {
    std::vector<std::vector<double>> bands;   // [trama][banda]
    std::vector<std::vector<double>> fine;    // [trama][bin]
    std::vector<double> total;
    std::vector<double> nu;          // grados de libertad por banda
    double df = 0.0;
    std::size_t nfft = 0;
};

FramePowers frame_powers(const Channels& x, double sr, const std::vector<double>& centers, double bw_oct,
                         double bin_hz, double fine_max_hz, std::size_t overlap) {
    const std::size_t n_in = x.empty() ? 0 : x[0].size();
    const std::size_t nfft = dsp::next_pow2(sr / bin_hz);
    const std::size_t n = std::max(n_in, nfft);
    // señal más corta que una trama: al centro, donde la ventana pesa; al principio quedaría donde
    // Hann vale casi cero y un golpe corto se leería como silencio
    const std::size_t left = n_in < nfft ? (nfft - n_in) / 2 : 0;
    const std::size_t hop = nfft / overlap;
    const auto win = dsp::hann_periodic(nfft);
    double wsum = 0.0;
    for (double w : win) wsum += w * w;
    const double scale = 1.0 / (double(nfft) * wsum);   // suma de bins = potencia media de la trama
    const double df = sr / double(nfft);
    const std::size_t n_bins = nfft / 2 + 1;

    // Integral exacta de la potencia entre los bordes de cada banda, interpolando dentro
    // del bin (el bin k cubre [k - 1/2, k + 1/2]·df).
    const std::size_t nb = centers.size();
    std::vector<std::size_t> i_lo(nb), i_hi(nb);
    std::vector<double> w_lo(nb), w_hi(nb), nu(nb);
    auto edge = [&](double f, std::size_t& i, double& w) {
        const double pos = std::clamp(f / df + 0.5, 0.0, double(n_bins));
        i = std::min(static_cast<std::size_t>(std::floor(pos)), n_bins - 1);
        w = pos - double(i);
    };
    for (std::size_t b = 0; b < nb; ++b) {
        const double lo = centers[b] * std::pow(2.0, -bw_oct / 2), hi = centers[b] * std::pow(2.0, bw_oct / 2);
        edge(lo, i_lo[b], w_lo[b]);
        edge(hi, i_hi[b], w_hi[b]);
        nu[b] = band_dof(i_lo[b], w_lo[b], i_hi[b], w_hi[b]);
    }
    const std::size_t n_fine = static_cast<std::size_t>(fine_max_hz / df) + 1;

    FramePowers out;
    out.nu = nu;
    out.df = df;
    out.nfft = nfft;
    dsp::RealFft fft(nfft);
    std::vector<double> frame(nfft), p(n_bins), c(n_bins + 1);
    std::vector<dsp::cplx> spec(n_bins);
    for (std::size_t start = 0; start + nfft <= n; start += hop) {
        std::fill(p.begin(), p.end(), 0.0);
        for (const auto& ch : x) {
            for (std::size_t i = 0; i < nfft; ++i) {
                const std::size_t k = start + i;
                frame[i] = (k >= left && k - left < n_in ? ch[k - left] : 0.0) * win[i];
            }
            fft.forward(frame.data(), nfft, spec.data());
            for (std::size_t k = 0; k < n_bins; ++k) p[k] += std::norm(spec[k]);
        }
        for (std::size_t k = 0; k < n_bins; ++k) p[k] *= scale * ((k > 0 && k + 1 < n_bins) ? 2.0 : 1.0);
        c[0] = 0.0;
        for (std::size_t k = 0; k < n_bins; ++k) c[k + 1] = c[k] + p[k];
        std::vector<double> bands(nb);
        for (std::size_t b = 0; b < nb; ++b)
            bands[b] = (c[i_hi[b]] + w_hi[b] * p[i_hi[b]]) - (c[i_lo[b]] + w_lo[b] * p[i_lo[b]]);
        out.bands.push_back(std::move(bands));
        out.fine.emplace_back(p.begin(), p.begin() + std::min(n_fine, n_bins));
        out.total.push_back(c[n_bins]);
    }
    return out;
}

}  // namespace

std::optional<Spectrum> long_term_spectrum(const Channels& x, double sr, double active_range_db,
                                           double floor_pct, double fine_max_hz) {
    Spectrum s;
    s.centers = band_centers(12.5, std::min(16000.0, 0.45 * sr));
    const std::size_t nb = s.centers.size();

    // Dos resoluciones. Tramas largas (~1.5 Hz por bin, 0.68 s a 48 kHz) para el nivel
    // medio y el espectro fino del hum. Tramas de ~170 ms para el piso: con 0.68 s un
    // patrón de batería llena todas las tramas y parece estacionario.
    const auto lng = frame_powers(x, sr, s.centers, 1.0 / 3.0, 1.5, fine_max_hz, 4);
    const auto sht = frame_powers(x, sr, s.centers, 1.0 / 3.0, 6.0, 0.0, 2);

    std::vector<double> total_db(lng.total.size());
    std::vector<double> valid_db;
    for (std::size_t t = 0; t < lng.total.size(); ++t) {
        total_db[t] = to_db(lng.total[t]);
        if (total_db[t] > kSilenceDb) valid_db.push_back(total_db[t]);
    }
    if (valid_db.empty()) return std::nullopt;
    const double gate = dsp::percentile(valid_db, 95.0) - active_range_db;

    std::vector<double> mean(nb, 0.0);
    std::size_t n_active = 0;
    for (std::size_t t = 0; t < lng.total.size(); ++t) {
        if (!(total_db[t] > kSilenceDb && total_db[t] >= gate)) continue;
        ++n_active;
        for (std::size_t b = 0; b < nb; ++b) mean[b] += lng.bands[t][b];
    }
    for (auto& v : mean) v /= double(n_active);
    s.active_fraction = double(n_active) / double(lng.total.size());

    // Piso estacionario por cuantil (Stahl, Fischer y Bippus, 2000; en la línea de la estadística
    // de mínimos de Martin, 2001): un percentil bajo de la potencia de cada banda en el tiempo,
    // dividido por el cociente percentil/media de una chi² con nu grados de libertad (band_dof).
    std::vector<std::size_t> short_valid;
    for (std::size_t t = 0; t < sht.total.size(); ++t)
        if (to_db(sht.total[t]) > kSilenceDb) short_valid.push_back(t);
    std::vector<double> floor(nb, 0.0);
    std::vector<double> col;
    for (std::size_t b = 0; b < nb; ++b) {
        col.clear();
        for (std::size_t t : short_valid) col.push_back(sht.bands[t][b]);
        const double nu = sht.nu[b];
        const double bias = dsp::chi2_ppf(floor_pct / 100.0, nu) / nu;
        const double pct = col.empty() ? std::nan("") : dsp::percentile(col, floor_pct);
        floor[b] = std::min(pct / bias, mean[b]);
    }

    double max_ratio = 0.0;
    for (std::size_t b = 0; b < nb; ++b)
        if (s.centers[b] >= 20.0) max_ratio = std::max(max_ratio, mean[b] / std::max(floor[b], 1e-30));
    s.stationary = max_ratio < 2.0;

    s.mean_db.resize(nb);
    s.floor_db.resize(nb);
    s.excess_db.resize(nb);
    for (std::size_t b = 0; b < nb; ++b) {
        s.mean_db[b] = to_db(mean[b]);
        s.floor_db[b] = to_db(floor[b]);
        const double ex = s.stationary ? mean[b] : std::max(mean[b] - floor[b], 1e-3 * mean[b]);
        s.excess_db[b] = to_db(ex);
    }

    s.frame_db.resize(sht.bands.size());
    for (std::size_t t = 0; t < sht.bands.size(); ++t) {
        s.frame_db[t].resize(nb);
        for (std::size_t b = 0; b < nb; ++b) s.frame_db[t][b] = to_db(sht.bands[t][b]);
    }
    s.frame_t0 = 0.5 * double(sht.nfft) / sr;
    s.frame_hop = double(sht.nfft / 2) / sr;

    const std::size_t n_fine = lng.fine.empty() ? 0 : lng.fine[0].size();
    s.fine_freqs.resize(n_fine);
    s.fine_floor_db.resize(n_fine);
    for (std::size_t k = 0; k < n_fine; ++k) {
        s.fine_freqs[k] = double(k) * lng.df;
        col.clear();
        for (std::size_t t = 0; t < lng.total.size(); ++t)
            if (total_db[t] > kSilenceDb) col.push_back(lng.fine[t][k]);
        s.fine_floor_db[k] = to_db(dsp::percentile(col, floor_pct));
    }
    return s;
}

double spectral_edge(const Spectrum& spec, double f_floor, double f_top, double range_db) {
    // Borde inferior de la región conectada al máximo (sin piso) que queda a menos de
    // range_db de él. Una isla grave separada por un valle más hondo no cuenta.
    const auto& level = spec.excess_db;
    const std::size_t peak = spec.peak_index(f_floor, f_top);
    const double thr = level[peak] - range_db;
    const std::size_t lo = static_cast<std::size_t>(
        std::lower_bound(spec.centers.begin(), spec.centers.end(), f_floor) - spec.centers.begin());
    std::size_t i = peak;
    while (i > lo && level[i - 1] >= thr) --i;
    if (i == lo) return spec.centers[lo];
    const double a = level[i - 1], b = level[i];
    const double t = (thr - a) / (b - a);
    return spec.centers[i - 1] * std::pow(spec.centers[i] / spec.centers[i - 1], t);
}

std::optional<HumFound> detect_hum(const Spectrum& spec, double prominence_db) {
    // Serie de 50 o 60 Hz en el piso estacionario. Las notas no caen en 50/60 Hz exactos
    // (G1 = 49.0, B1 = 61.7 Hz con A4 = 440), y una nota no forma parte del piso.
    const auto& f = spec.fine_freqs;
    const auto& db = spec.fine_floor_db;
    if (f.size() < 8) return std::nullopt;
    const double df = f[1] - f[0];
    std::optional<HumFound> best;
    for (double base : {50.0, 60.0}) {
        std::vector<double> hits;
        for (int k = 1; k < 9; ++k) {
            const double fk = k * base;
            if (fk * std::pow(2.0, 1.0 / 6.0) > f.back()) break;
            // máximo local a menos de 1 Hz de k·base; sin esto la falda de un parcial pasa por pico
            const long i = static_cast<long>(std::nearbyint(fk / df));
            long j = i - 2;
            for (long q = i - 2; q < i + 3; ++q)
                if (db[q] > db[j]) j = q;
            std::vector<double> ctx;
            for (std::size_t q = 0; q < f.size(); ++q) {
                const bool in = f[q] > fk * std::pow(2.0, -1.0 / 6.0) && f[q] < fk * std::pow(2.0, 1.0 / 6.0);
                const bool near = f[q] > fk - 4 && f[q] < fk + 4;
                if (in && !near) ctx.push_back(db[q]);
            }
            if (std::abs(f[j] - fk) < 1.0 && db[j] > db[j - 1] && db[j] > db[j + 1] &&
                db[j] - dsp::median(ctx) >= prominence_db)
                hits.push_back(fk);
        }
        // serie creíble: incluye la fundamental o el segundo armónico (rectificación)
        const bool has_base = std::find(hits.begin(), hits.end(), base) != hits.end();
        const bool has_two = std::find(hits.begin(), hits.end(), 2 * base) != hits.end();
        const bool credible = hits.size() >= 2 && (has_base || has_two);
        if (credible && (!best || hits.size() > best->harmonics_hz.size()))
            best = HumFound{base, hits, 0.0};
    }
    if (!best) return std::nullopt;
    double level = -1e300;
    for (double h : best->harmonics_hz) {
        double sum = 0.0;
        for (std::size_t q = 0; q < f.size(); ++q)
            if (f[q] > h - 1.5 && f[q] < h + 1.5) sum += std::pow(10.0, db[q] / 10.0);
        level = std::max(level, 10.0 * std::log10(sum));
    }
    best->level_db = level;
    return best;
}

}  // namespace hpf::analysis
