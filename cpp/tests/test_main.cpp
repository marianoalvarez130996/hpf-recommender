// Pruebas del port en C++. Las mismas señales con respuesta conocida que tests/ en Python,
// más las piezas numéricas (FFT, remuestreo, Butterworth, chi²) contra valores de scipy.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "analysis.h"
#include "design.h"
#include "dsp.h"
#include "hpf/hpf.h"
#include "pitch.h"
#include "wav.h"

namespace {

int g_failed = 0, g_checks = 0;
std::string g_current;

void check(bool ok, const char* expr, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failed;
        std::printf("  FAIL %s (line %d): %s\n", g_current.c_str(), line, expr);
    }
}
#define CHECK(e) check((e), #e, __LINE__)
#define NEAR(a, b, tol) check(std::abs((a) - (b)) <= (tol), #a " ~ " #b, __LINE__)

std::vector<std::pair<std::string, std::function<void()>>>& tests() {
    static std::vector<std::pair<std::string, std::function<void()>>> t;
    return t;
}
struct Reg {
    Reg(const char* n, std::function<void()> f) { tests().emplace_back(n, std::move(f)); }
};
#define TEST(name) \
    void name();   \
    Reg reg_##name(#name, name); \
    void name()

constexpr double SR = 48000.0;
constexpr double PI = 3.14159265358979323846;
constexpr double E1 = 41.20, E2 = 82.41, B2 = 123.47;

using Sig = std::vector<double>;

Sig harmonic(double f0, double dur, int n = 12, double fund_db = 0.0, double amp = 0.3) {
    const std::size_t len = std::size_t(dur * SR);
    Sig x(len, 0.0);
    for (std::size_t i = 0; i < len; ++i) {
        const double t = double(i) / SR;
        double v = 0.0;
        for (int k = 2; k <= n; ++k)
            if (k * f0 < 0.45 * SR) v += std::sin(2 * PI * k * f0 * t + k) / k;
        v += std::pow(10.0, fund_db / 20.0) * std::sin(2 * PI * f0 * t);
        x[i] = v;
    }
    double peak = 0.0;
    for (double v : x) peak = std::max(peak, std::abs(v));
    for (auto& v : x) v *= amp / peak;
    return x;
}

Sig only_harmonics(double f0, int k0, int k1, double dur, double amp = 0.3) {
    const std::size_t len = std::size_t(dur * SR);
    Sig x(len, 0.0);
    double peak = 0.0;
    for (std::size_t i = 0; i < len; ++i) {
        const double t = double(i) / SR;
        for (int k = k0; k <= k1; ++k) x[i] += std::sin(2 * PI * k * f0 * t + k) / k;
        peak = std::max(peak, std::abs(x[i]));
    }
    for (auto& v : x) v *= amp / peak;
    return x;
}

Sig band_noise(std::size_t n, double lo, double hi, int order, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g;
    Sig x(n);
    for (auto& v : x) v = g(rng);
    hpf::dsp::sosfilt(hpf::dsp::butter_sos(order, lo, SR, true), x.data(), n);
    hpf::dsp::sosfilt(hpf::dsp::butter_sos(order, hi, SR, false), x.data(), n);
    return x;
}

Sig bursts(double dur, unsigned seed = 0) {
    const std::size_t n = std::size_t(dur * SR);
    Sig x = band_noise(n, 200.0, 8000.0, 4, seed);
    Sig env(n, 0.0);
    const std::size_t len = std::size_t(5 * 0.05 * SR), every = std::size_t(0.25 * SR);
    for (std::size_t s = 0; s + len < n; s += every)
        for (std::size_t i = 0; i < len; ++i) env[s + i] += std::exp(-double(i) / SR / 0.05);
    double peak = 0.0;
    for (std::size_t i = 0; i < n; ++i) { x[i] *= env[i]; peak = std::max(peak, std::abs(x[i])); }
    for (auto& v : x) v *= 0.3 / peak;
    return x;
}

Sig rumble(double dur, double f, double rms, unsigned seed = 1) {
    Sig x = band_noise(std::size_t(dur * SR), f / 1.3, f * 1.3, 2, seed);
    double e = 0.0;
    for (double v : x) e += v * v;
    const double s = rms / std::sqrt(e / double(x.size()));
    for (auto& v : x) v *= s;
    return x;
}

Sig add(Sig a, const Sig& b) {
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) a[i] += b[i];
    return a;
}
Sig cat(Sig a, const Sig& b) { a.insert(a.end(), b.begin(), b.end()); return a; }
hpf::Channels mono(const Sig& x) { return {x}; }
double cents(double a, double b) { return std::abs(1200.0 * std::log2(a / b)); }

// ---------------------------------------------------------------- piezas numéricas

TEST(fft_matches_direct_dft) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-1, 1);
    // largos con log2(n/2) par e impar: la FFT compleja interna cambia de etapas
    for (std::size_t n : {4u, 8u, 32u, 64u, 1024u, 2048u}) {
        Sig x(n);
        for (auto& v : x) v = u(rng);
        hpf::dsp::RealFft f(n);
        std::vector<std::complex<double>> X(n / 2 + 1);
        f.forward(x.data(), n, X.data());
        double err = 0.0;
        for (std::size_t k = 0; k <= n / 2; ++k) {
            std::complex<double> s = 0.0;
            for (std::size_t m = 0; m < n; ++m) s += x[m] * std::polar(1.0, -2 * PI * double(k * m) / double(n));
            err = std::max(err, std::abs(s - X[k]));
        }
        CHECK(err < 1e-9 * double(n));
        Sig back(n);
        f.inverse(X.data(), back.data());
        double e2 = 0.0;
        for (std::size_t m = 0; m < n; ++m) e2 = std::max(e2, std::abs(back[m] - x[m]));
        CHECK(e2 < 1e-12 * double(n));
    }
}

TEST(complex_fft_matches_direct_dft) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u(-1, 1);
    for (std::size_t n : {1u, 2u, 4u, 8u, 16u, 128u, 512u}) {
        std::vector<std::complex<double>> x(n), y;
        for (auto& v : x) v = {u(rng), u(rng)};
        y = x;
        hpf::dsp::Fft f(n);
        f.forward(y.data());
        double err = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            std::complex<double> s = 0.0;
            for (std::size_t m = 0; m < n; ++m) s += x[m] * std::polar(1.0, -2 * PI * double(k * m) / double(n));
            err = std::max(err, std::abs(s - y[k]));
        }
        CHECK(err < 1e-12 * double(n + 1));
        f.inverse(y.data());
        double e2 = 0.0;
        for (std::size_t m = 0; m < n; ++m) e2 = std::max(e2, std::abs(y[m] - x[m]));
        CHECK(e2 < 1e-13 * double(n + 1));
    }
}

TEST(real_fft_zero_pads_short_input) {
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(-1, 1);
    const std::size_t n = 256, len = 173;
    Sig x(len);
    for (auto& v : x) v = u(rng);
    hpf::dsp::RealFft f(n);
    std::vector<std::complex<double>> X(n / 2 + 1);
    f.forward(x.data(), len, X.data());
    double err = 0.0;
    for (std::size_t k = 0; k <= n / 2; ++k) {
        std::complex<double> s = 0.0;
        for (std::size_t m = 0; m < len; ++m) s += x[m] * std::polar(1.0, -2 * PI * double(k * m) / double(n));
        err = std::max(err, std::abs(s - X[k]));
    }
    CHECK(err < 1e-10);
}

TEST(resample_poly_matches_scipy) {
    Sig x(4800);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(2 * PI * 1000.0 * double(i) / 48000.0);
    const auto y = hpf::dsp::resample_poly(x, 1, 3);
    CHECK(y.size() == 1600);
    NEAR(y[100], 1.0009837275316082, 1e-12);
    NEAR(y[777], -0.3830598885933605, 1e-12);
    const auto y2 = hpf::dsp::resample_poly(x, 160, 441);
    CHECK(y2.size() == 1742);
    NEAR(y2[100], -1.0000156473164992, 1e-12);
    NEAR(y2[1000], 0.4719946907090686, 1e-12);
}

TEST(butter_matches_scipy) {
    const auto s = hpf::dsp::butter_sos(3, 80.0, 48000.0, true);
    CHECK(s.size() == 2);
    NEAR(s[0][0], 0.98958262, 1e-8);
    NEAR(s[0][4], -0.98958248, 1e-8);
    NEAR(s[1][4], -1.98947367, 1e-8);
    NEAR(s[1][5], 0.98958276, 1e-8);
    for (int order = 1; order <= 8; ++order) {
        const auto h = hpf::dsp::butter_sos(order, 100.0, 48000.0, true);
        NEAR(20 * std::log10(std::abs(hpf::dsp::sosfreqz(h, 100.0, 48000.0))), -3.0103, 1e-3);
        NEAR(-20 * std::log10(std::abs(hpf::dsp::sosfreqz(h, 10.0, 48000.0))), hpf::design::attenuation_db(10.0, 100.0, order), 0.05);
        const auto l = hpf::dsp::butter_sos(order, 5000.0, 48000.0, false);
        NEAR(20 * std::log10(std::abs(hpf::dsp::sosfreqz(l, 5000.0, 48000.0))), -3.0103, 1e-3);
    }
}

TEST(chi2_ppf_matches_scipy) {
    NEAR(hpf::dsp::chi2_ppf(0.1, 2), 0.21072103131565273, 1e-10);
    NEAR(hpf::dsp::chi2_ppf(0.1, 10), 4.865182051925328, 1e-9);
    NEAR(hpf::dsp::chi2_ppf(0.5, 1), 0.454936423119572, 1e-10);
    NEAR(hpf::dsp::chi2_ppf(0.1, 5.2), 1.725769441520292, 1e-9);
    NEAR(hpf::dsp::chi2_ppf(0.1, 52.7), 40.04521892601933, 1e-8);
    NEAR(hpf::dsp::chi2_ppf(0.9, 3), 6.251388631170325, 1e-9);
}

TEST(percentile_and_median_like_numpy) {
    const std::vector<double> v = {3, 1, 4, 1, 5, 9, 2, 6};
    NEAR(hpf::dsp::percentile(v, 10), 1.0, 1e-12);
    NEAR(hpf::dsp::percentile(v, 95), 7.95, 1e-12);
    NEAR(hpf::dsp::median(v), 3.5, 1e-12);
    NEAR(hpf::dsp::median({5, 1, 3}), 3.0, 1e-12);
}

// ---------------------------------------------------------------- diseño

TEST(max_cutoff_gives_exact_loss) {
    for (int order : {1, 2, 3, 4, 6, 8})
        for (double loss : {0.5, 1.0, 3.0}) {
            const double fc = hpf::design::max_cutoff(80.0, order, loss);
            NEAR(hpf::design::attenuation_db(80.0, fc, order), loss, 1e-9);
        }
}

TEST(digital_filter_matches_analog_below_1k) {
    for (double sr : {44100.0, 48000.0, 96000.0})
        for (int order : {2, 4})
            for (double keep : {30.0, 82.4, 400.0}) {
                const double fc = hpf::design::max_cutoff(keep, order, 1.0);
                const auto r = hpf::design::response_at(keep, fc, order, sr);
                NEAR(r.loss_db, 1.0, 0.02);
                CHECK(r.phase_deg > 0 && r.group_delay_ms > 0);
            }
}

TEST(snap_down_r20) {
    CHECK(hpf::design::snap_down(100.0) == 100.0);
    CHECK(hpf::design::snap_down(99.9) == 90.0);
    for (int i = 0; i < 50; ++i) {
        const double fc = 20.0 * std::pow(900.0 / 20.0, i / 49.0);
        const double s = hpf::design::snap_down(fc);
        CHECK(s <= fc && s > fc / 1.13);
    }
}

TEST(choose_takes_lowest_order_that_rejects) {
    const auto c = hpf::design::choose(100.0, {25.0}, {20.0}, 1.0, {2, 3, 4}, 16, 300, true, 0.0);
    CHECK(c.met && c.short_db == 0.0);
    CHECK(hpf::design::attenuation_db(25.0, c.fc, c.order) >= 20.0);
    const double prev = hpf::design::snap_down(hpf::design::max_cutoff(100.0, c.order - 1, 1.0));
    CHECK(hpf::design::attenuation_db(25.0, prev, c.order - 1) < 20.0);
}

TEST(choose_reports_shortfall) {
    const auto c = hpf::design::choose(100.0, {70.0}, {40.0}, 1.0, {2, 3, 4}, 16, 300);
    CHECK(!c.met && c.order == 4 && c.short_db > 1.0);
}

// ---------------------------------------------------------------- nota más grave

TEST(bass_low_e) {
    const auto r = hpf::pitch::estimate_min_f0(mono(harmonic(E1, 2.0)), SR);
    CHECK(r.f0_min && cents(*r.f0_min, E1) < 20);
    CHECK(r.voiced_energy > 0.9);
}

TEST(weak_fundamental_is_not_an_octave_up) {
    const auto r = hpf::pitch::estimate_min_f0(mono(harmonic(E1, 2.0, 12, -20.0)), SR);
    CHECK(r.f0_min && cents(*r.f0_min, E1) < 20);
}

TEST(power_chord_gives_lowest_note) {
    const auto r = hpf::pitch::estimate_min_f0(mono(add(harmonic(E2, 2.0), harmonic(B2, 2.0))), SR);
    CHECK(r.f0_min && cents(*r.f0_min, E2) < 20);
}

TEST(missing_fundamental_protects_lowest_partial) {
    const auto r = hpf::pitch::estimate_min_f0(mono(only_harmonics(55.0, 2, 8, 2.0)), SR);
    CHECK(r.f0_min && cents(*r.f0_min, 110.0) < 20);
}

TEST(rare_low_note_is_kept) {
    const auto r = hpf::pitch::estimate_min_f0(mono(cat(harmonic(E2, 3.0), harmonic(E1, 0.5))), SR);
    CHECK(r.f0_min && cents(*r.f0_min, E1) < 20);
}

TEST(noise_has_no_note) {
    std::mt19937_64 rng(3);
    std::normal_distribution<double> g;
    Sig x(std::size_t(3 * SR));
    for (auto& v : x) v = 0.1 * g(rng);
    const auto r = hpf::pitch::estimate_min_f0(mono(x), SR);
    CHECK(!r.f0_min);
    CHECK(r.voiced_energy < 0.05);
}

// ---------------------------------------------------------------- recomendación

TEST(silence) {
    const auto r = hpf::recommend(mono(Sig(std::size_t(2 * SR), 0.0)), SR);
    CHECK(!r.cutoff_hz && !r.warnings.empty());
}

TEST(nan_is_rejected) {
    Sig x(std::size_t(SR), 0.0);
    x[100] = std::nan("");
    bool threw = false;
    try { hpf::recommend(mono(x), SR); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

TEST(bass_with_subsonic_rumble) {
    const auto r = hpf::recommend(mono(add(harmonic(E1, 6.0), rumble(6.0, 16.0, 0.01))), SR);
    CHECK(r.source == "f0");
    CHECK(r.keep_hz && std::abs(*r.keep_hz - E1) < 1.0);
    CHECK(r.loss_budget_db && *r.loss_budget_db == 1.0);   // una nota se protege con 1 dB
    CHECK(r.loss_at_keep_db && *r.loss_at_keep_db <= 1.02);
    CHECK(r.active && r.rejection_met);     // el rumble subsónico queda en el objetivo
}

bool warns(const hpf::Recommendation& r, const std::string& text) {
    return std::any_of(r.warnings.begin(), r.warnings.end(),
                       [&](const std::string& w) { return w.find(text) != std::string::npos; });
}

TEST(percussion_edge_clear_of_stationary_noise) {
    const auto r = hpf::recommend(mono(add(bursts(8.0), rumble(8.0, 16.0, 0.001))), SR);
    CHECK(r.source == "espectro");
    CHECK(r.keep_hz && *r.keep_hz > 100.0 && *r.keep_hz < 220.0);
    // sin altura y con el borde libre de ruido, el corte va en el borde (3 dB); el redondeo R20
    // hacia abajo deja algo menos
    CHECK(r.loss_budget_db && *r.loss_budget_db == 3.0);
    CHECK(*r.loss_at_keep_db > 1.0 && *r.loss_at_keep_db <= 3.02);
    CHECK(!warns(r, "reaches the spectral edge"));
    CHECK(r.active && r.rejection_met);       // el rumble de 16 Hz queda en el objetivo
}

TEST(percussion_edge_reached_by_stationary_noise) {
    // la falda del rumble de 30 Hz cubre la de las ráfagas: restar el piso sube el borde medido,
    // y ese borde se protege como una nota
    const auto r = hpf::recommend(mono(add(bursts(8.0), rumble(8.0, 30.0, 0.02))), SR);
    CHECK(r.source == "espectro");
    CHECK(r.keep_hz && *r.keep_hz > 100.0 && *r.keep_hz < 220.0);
    CHECK(*r.cutoff_hz < *r.keep_hz);
    CHECK(r.loss_budget_db && *r.loss_budget_db == 1.0);
    CHECK(*r.loss_at_keep_db <= 1.02);
    CHECK(warns(r, "reaches the spectral edge"));
}

TEST(edge_slope_stops_at_24_db_per_octave) {
    // el borde medido tiene error; con más pendiente se volvería pérdida. Como ningún orden hasta 4
    // cumple, sin el tope se habría elegido 48 dB/oct. Fijando el orden sí se puede.
    const auto x = mono(add(bursts(8.0), rumble(8.0, 60.0, 0.03)));
    const auto r = hpf::recommend(x, SR);
    CHECK(r.source == "espectro" && r.order <= 4);
    CHECK(!r.rejection_met && warns(r, "no steeper slope is used"));
    hpf::Params p;
    p.order = 8;
    CHECK(hpf::recommend(x, SR, p).order == 8);
}

TEST(nothing_to_remove_leaves_the_filter_off) {
    // sin ruido grave ni offset DC el filtro no quitaría nada medible: se recomienda apagado, con el
    // corte máximo seguro por si igual se quiere filtrar
    const auto r = hpf::recommend(mono(harmonic(220.0, 6.0)), SR);
    CHECK(!r.active);
    CHECK(r.cutoff_hz && r.loss_at_keep_db && *r.loss_at_keep_db <= 1.02);
    CHECK(warns(r, "would remove nothing"));
}

TEST(dc_offset_turns_the_filter_on_without_the_fixed_filter) {
    Sig x = harmonic(220.0, 6.0);
    for (auto& v : x) v += 0.01;
    hpf::Params p;
    p.base_hz = 0.0;
    const auto r = hpf::recommend(mono(x), SR, p);
    CHECK(r.active && !r.base_hz && r.base_order == 0);
    CHECK(!warns(r, "would remove nothing"));
}

TEST(fixed_20hz_filter_removes_the_dc_offset) {
    // el filtro fijo de 20 Hz, siempre puesto, ya quita el offset: no hace falta otro filtro
    Sig x = harmonic(220.0, 6.0);
    for (auto& v : x) v += 0.01;
    const auto r = hpf::recommend(mono(x), SR);
    CHECK(r.base_hz && *r.base_hz == 20.0 && r.base_order == 4);
    CHECK(!r.active);
    CHECK(warns(r, "fixed 20 Hz filter removes it"));
}

TEST(infrasonic_rumble_is_left_to_the_fixed_filter) {
    // sin el filtro fijo este rumble de 8 Hz enciende el filtro (el ruido de estas señales no es el
    // de las de Python: por eso el nivel es otro)
    const auto r = hpf::recommend(mono(add(harmonic(E1, 6.0), rumble(6.0, 8.0, 0.01))), SR);
    CHECK(r.keep_hz && std::abs(*r.keep_hz - E1) < 1.0);
    CHECK(!r.active);
}

TEST(infrasonic_rumble_does_not_pull_the_protected_note_down) {
    const auto r = hpf::recommend(mono(add(harmonic(E1, 6.0), rumble(6.0, 8.0, 0.1))), SR);
    CHECK(r.source == "f0" && r.keep_hz && std::abs(*r.keep_hz - E1) < 1.0);
}

TEST(fixed_and_recommended_filters_share_the_budget) {
    // el recomendado va detrás del fijo: entre los dos no pasan de 1 dB en la nota
    for (double f0 : {30.87, 29.14}) {
        const auto r = hpf::recommend(mono(add(harmonic(f0, 6.0), rumble(6.0, 16.0, 0.03))), SR);
        CHECK(r.keep_hz && std::abs(*r.keep_hz - f0) < 1.0 && r.active);
        const double chain = hpf::design::attenuation_db(*r.keep_hz, 20.0, 4) +
                             hpf::design::attenuation_db(*r.keep_hz, *r.cutoff_hz, r.order);
        CHECK(chain <= 1.02 && r.loss_at_keep_db && *r.loss_at_keep_db <= 1.02);
    }
}

TEST(gentlest_slope_that_meets_the_target) {
    // un rumble débil y lejos de la nota se va con 6 dB/oct
    const auto r = hpf::recommend(mono(add(harmonic(220.0, 6.0), rumble(6.0, 16.0, 0.01))), SR);
    CHECK(r.active && r.rejection_met);
    CHECK(r.order == 1 && r.slope_db_oct == 6);
}

TEST(steeper_than_24_db_per_octave_when_needed) {
    // rumble fuerte poco más de media octava bajo E2: con 24 dB/oct no alcanza, con más pendiente sí
    const auto x = mono(add(harmonic(E2, 6.0), rumble(6.0, 45.0, 0.03)));
    const auto r = hpf::recommend(x, SR);
    CHECK(r.source == "f0" && r.keep_hz && std::abs(*r.keep_hz - E2) < 1.0);
    CHECK(r.order > 4 && r.rejection_met);
    CHECK(r.loss_at_keep_db && *r.loss_at_keep_db <= 1.02);
    hpf::Params p;
    p.max_order = 4;
    CHECK(!hpf::recommend(x, SR, p).rejection_met);
}

TEST(hum_series_is_reported) {
    Sig hum(std::size_t(8 * SR));
    for (std::size_t i = 0; i < hum.size(); ++i)
        for (int k = 1; k <= 3; ++k) hum[i] += 0.003 / k * std::sin(2 * PI * 60.0 * k * double(i) / SR);
    const auto r = hpf::recommend(mono(add(bursts(8.0), hum)), SR);
    CHECK(r.hum && r.hum->base_hz == 60.0);
    CHECK(r.hum && std::find(r.hum->harmonics_hz.begin(), r.hum->harmonics_hz.end(), 60.0) != r.hum->harmonics_hz.end());
}

TEST(antiphase_stereo_is_not_cancelled) {
    const Sig s = add(bursts(6.0), harmonic(110.0, 6.0));
    Sig neg = s;
    for (auto& v : neg) v = -v;
    const auto m = hpf::recommend(mono(s), SR);
    const auto st = hpf::recommend({s, neg}, SR);
    CHECK(st.cutoff_hz && m.cutoff_hz && *st.cutoff_hz == *m.cutoff_hz);
    CHECK(std::abs(*st.keep_hz - *m.keep_hz) <= 0.02 * *m.keep_hz);
}

TEST(short_signal_does_not_crash) {
    const auto r = hpf::recommend(mono(harmonic(110.0, 0.3)), SR);
    bool found = false;
    for (const auto& w : r.warnings) found |= w.find("3 s") != std::string::npos;
    CHECK(found);
}

Sig line(const std::vector<double>& notes, int reps, double dur) {
    Sig x;
    for (int r = 0; r < reps; ++r)
        for (double f : notes) x = cat(x, harmonic(f, dur));
    return x;
}

TEST(narrowband_rumble_is_not_a_note) {
    const Sig l = line({E2, 98.0, 110.0, 146.8, 123.5, 98.0}, 2, 0.6);
    const auto r = hpf::recommend(mono(add(l, rumble(double(l.size()) / SR, 45.0, 0.02))), SR);
    CHECK(r.keep_hz && cents(*r.keep_hz, E2) < 50);
}

TEST(drone_is_kept) {
    const double dur = 7.2;
    const Sig pedal = harmonic(E1, dur, 12, 0.0, 0.15);
    const Sig moving = line({E2 * 2, 196.0, 220.0, 246.9}, 3, dur / 12);
    const auto r = hpf::recommend(mono(add(add(pedal, moving), rumble(dur, 22.0, 0.005))), SR);
    CHECK(r.keep_hz && cents(*r.keep_hz, E1) < 50);
}

TEST(wav_roundtrip_keeps_format) {
    const Sig a = harmonic(E1, 1.0), b = harmonic(E1 * 1.5, 1.0);
    const std::string path = (std::filesystem::temp_directory_path() / "hpf_rt.wav").string();
    for (auto fmt : {hpf::wav::Format{1, 8, 2, 48000}, hpf::wav::Format{1, 16, 2, 44100},
                     hpf::wav::Format{1, 24, 2, 48000}, hpf::wav::Format{1, 32, 2, 96000},
                     hpf::wav::Format{3, 32, 2, 48000}, hpf::wav::Format{3, 64, 2, 48000}}) {
        hpf::wav::write(path, {a, b}, fmt);
        hpf::wav::Format got;
        const auto x = hpf::wav::read(path, got);
        CHECK(got.tag == fmt.tag && got.bits == fmt.bits && got.channels == 2 && got.sample_rate == fmt.sample_rate);
        double err = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(x[0][i] - a[i]));
        CHECK(err <= 1.0 / std::ldexp(1.0, fmt.bits - 1) + 1e-7);
    }
    std::filesystem::remove(path);
}

TEST(apply_hpf_meets_reported_loss) {
    // senoidal en keep_hz a través del filtro: la pérdida medida es la que se reporta
    const auto r = hpf::recommend(mono(harmonic(E1, 6.0)), SR);
    Sig s(std::size_t(4 * SR));
    for (std::size_t i = 0; i < s.size(); ++i) s[i] = std::sin(2 * PI * *r.keep_hz * double(i) / SR);
    hpf::Channels c = {s};
    hpf::apply_hpf(c, SR, *r.cutoff_hz, r.order);
    double e_in = 0.0, e_out = 0.0;
    for (std::size_t i = std::size_t(SR); i < s.size(); ++i) { e_in += s[i] * s[i]; e_out += c[0][i] * c[0][i]; }
    NEAR(10 * std::log10(e_in / e_out), *r.loss_at_keep_db, 0.05);
}

// ---------------------------------------------------------------- regresiones

Sig hum60(std::size_t n, double rms) {
    Sig h(n, 0.0);
    double e = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (int k = 1; k <= 3; ++k) h[i] += std::sin(2 * PI * 60.0 * k * double(i) / SR) / k;
        e += h[i] * h[i];
    }
    const double s = rms / std::sqrt(e / double(n));
    for (auto& v : h) v *= s;
    return h;
}

double rms_of(const Sig& x) {
    double e = 0.0;
    for (double v : x) e += v * v;
    return std::sqrt(e / double(x.size()));
}

void write_bytes(const std::string& path, const std::string& bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
}

std::string le32(uint32_t v) { return std::string{char(v & 0xFF), char((v >> 8) & 0xFF), char((v >> 16) & 0xFF), char(v >> 24)}; }

TEST(wav_rejects_truncated_fmt) {
    // fmt declara 16 bytes y el archivo trae 4: lo que falta no se puede leer
    const std::string dir = std::filesystem::temp_directory_path().string();
    const std::string a = dir + "/hpf_fmt_trunc.wav", b = dir + "/hpf_ext_trunc.wav", c = dir + "/hpf_data_fmt_trunc.wav";
    write_bytes(a, "RIFF" + le32(20) + "WAVE" + "fmt " + le32(16) + std::string("\x01\x00\x01\x00", 4));
    std::string ext("\xFE\xFF\x01\x00\x80\xBB\x00\x00\x80\x32\x02\x00\x03\x00\x18\x00\x16\x00\x18\x00\x04\x00\x00\x00", 24);
    write_bytes(b, "RIFF" + le32(36) + "WAVE" + "fmt " + le32(40) + ext);
    write_bytes(c, "RIFF" + le32(40) + "WAVE" + "data" + le32(6) + std::string(6, '\0') + "fmt " + le32(16) + std::string("\x01\x00", 2));
    for (const auto& path : {a, b, c}) {
        bool ok = false;
        try {
            hpf::wav::Format f;
            hpf::wav::read(path, f);
        } catch (const std::runtime_error& e) {
            ok = std::string(e.what()).find("fmt incompleto") != std::string::npos;
        }
        CHECK(ok);
        std::filesystem::remove(path);
    }
}

TEST(floor_subtraction_recovers_intermittent_content) {
    // ruido estacionario + golpes 200 Hz-8 kHz + tono de 31.5 Hz el 10 % del tiempo que aporta a su
    // banda 1.5 veces lo del ruido: el exceso (media - piso) tiene que ser todo lo que no es ruido
    const std::size_t n = std::size_t(120 * SR);
    std::mt19937_64 rng(4);
    std::normal_distribution<double> g;
    Sig noise(n), mix(n);
    const Sig music = bursts(120.0, 4);
    for (std::size_t i = 0; i < n; ++i) {
        noise[i] = 0.02 * g(rng);
        const double t = double(i) / SR, m = std::fmod(t, 10.0);
        const double ramp = m < 1.0 ? std::clamp(std::min(m, 1.0 - m) / 0.05, 0.0, 1.0) : 0.0;
        mix[i] = noise[i] + 1.9e-3 * std::sin(2 * PI * 31.5 * t) * ramp + music[i];
    }
    const auto sm = hpf::analysis::long_term_spectrum({mix}, SR);
    const auto so = hpf::analysis::long_term_spectrum({noise}, SR);
    std::size_t b = 0;
    for (std::size_t k = 0; k < sm->centers.size(); ++k)
        if (std::abs(std::log(sm->centers[k] / 31.5)) < std::abs(std::log(sm->centers[b] / 31.5))) b = k;
    CHECK(!sm->stationary);
    const double expected = 10 * std::log10(std::pow(10.0, sm->mean_db[b] / 10) - std::pow(10.0, so->mean_db[b] / 10));
    NEAR(expected - so->mean_db[b], 1.76, 0.6);   // el caso está bien armado
    NEAR(sm->excess_db[b], expected, 1.5);
}

TEST(rare_low_note_over_hum_is_kept) {
    // G#1 0.3 s de 9.3 s, fundamental 17 dB abajo, hum de 60 Hz a -30 dB: mismo caso que en Python
    const double low = 51.91;
    Sig x;
    const int ks[] = {14, 16, 3, 16, 10, 11, 13, 7, 19, 3, 7, 9};
    for (int i = 0; i < 12; ++i) {
        if (i == 6) x = cat(x, harmonic(low, 0.3, 12, -17.0));
        x = cat(x, harmonic(low * std::pow(2.0, ks[i] / 12.0), 0.75, 12, -17.0));
    }
    const auto r = hpf::recommend(mono(add(x, hum60(x.size(), std::pow(10.0, -30.0 / 20.0) * rms_of(x)))), SR);
    CHECK(r.keep_hz && cents(*r.keep_hz, low) < 50);
    CHECK(r.cutoff_hz && hpf::design::attenuation_db(low, *r.cutoff_hz, r.order) <= 1.02);
}

TEST(short_burst_is_not_silence) {
    Sig x(100);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = 0.3 * std::sin(2 * PI * 1000.0 * double(i) / SR);
    const auto r = hpf::recommend(mono(x), SR);
    CHECK(r.cutoff_hz.has_value());
    bool silent = false;
    for (const auto& w : r.warnings) silent |= w.find("Digital silence") != std::string::npos;
    CHECK(!silent);
}

TEST(silence_json_has_params_and_dc) {
    const auto j = hpf::to_json(hpf::recommend(mono(Sig(std::size_t(2 * SR), 0.0)), SR));
    CHECK(j.find("\"params\": {\"max_loss_db\": 1.0, \"edge_loss_db\": 3.0, \"range_db\": 30.0, \"fmin\": 27.5, "
                 "\"fmax\": 1000.0, \"max_order\": 8, \"fc_max\": 300.0, \"base_hz\": 20.0}") != std::string::npos);
    CHECK(j.find("\"dc_offset_dbfs\": [-240.0]") != std::string::npos);
}

TEST(invalid_params_throw) {
    const auto x = mono(harmonic(E1, 1.0));
    auto bad = [&](std::function<void(hpf::Params&)> set) {
        hpf::Params p;
        set(p);
        try { hpf::recommend(x, SR, p); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    CHECK(bad([](hpf::Params& p) { p.fmin = 0.0; }));
    CHECK(bad([](hpf::Params& p) { p.fmax = 20.0; }));
    CHECK(bad([](hpf::Params& p) { p.fc_max = 10.0; }));
    CHECK(bad([](hpf::Params& p) { p.max_order = 0; }));
    CHECK(bad([](hpf::Params& p) { p.max_order = 9; }));
    CHECK(bad([](hpf::Params& p) { p.order = 9; }));
    CHECK(bad([](hpf::Params& p) { p.max_loss_db = -1.0; }));
    CHECK(bad([](hpf::Params& p) { p.edge_loss_db = 0.0; }));
    CHECK(bad([](hpf::Params& p) { p.order = -2; }));
}

TEST(json_numbers_like_python_repr) {
    // valores del reporte tal como los imprime json.dumps de Python
    hpf::Recommendation r;
    r.cutoff_hz = 1000.0;
    r.keep_hz = 0.0001;
    r.loss_at_keep_db = 1e-05;
    r.phase_at_keep_deg = -0.0;
    r.dc_offset_dbfs = {-240.0, 41.5, 0.153, 123456789012345680.0};
    const auto j = hpf::to_json(r);
    CHECK(j.find("\"cutoff_hz\": 1000.0,") != std::string::npos);
    CHECK(j.find("\"keep_hz\": 0.0001,") != std::string::npos);
    CHECK(j.find("\"loss_at_keep_db\": 1e-05,") != std::string::npos);
    CHECK(j.find("\"phase_at_keep_deg\": -0.0,") != std::string::npos);
    CHECK(j.find("[-240.0, 41.5, 0.153, 1.2345678901234568e+17]") != std::string::npos);
}

TEST(low_antiphase_bass_is_kept) {
    // platos en fase y un bajo en contrafase entre L y R, 12 dB abajo: sin la diferencia L-R el corte sube a 280 Hz
    const std::size_t n = std::size_t(6.0 * SR);
    Sig hats = band_noise(n, 2000.0, 12000.0, 4, 0);
    Sig env(n, 0.0);
    const std::size_t len = std::size_t(5 * 0.05 * SR), every = std::size_t(0.25 * SR);
    for (std::size_t s = 0; s + len < n; s += every)
        for (std::size_t i = 0; i < len; ++i) env[s + i] += std::exp(-double(i) / SR / 0.05);
    double peak = 0.0;
    for (std::size_t i = 0; i < n; ++i) { hats[i] *= env[i]; peak = std::max(peak, std::abs(hats[i])); }
    for (auto& v : hats) v *= 0.3 / peak;
    Sig bass = harmonic(E1, 6.0);
    for (auto& v : bass) v *= std::pow(10.0, -12.0 / 20.0);
    Sig l = hats, r = hats;
    for (std::size_t i = 0; i < n; ++i) { l[i] += bass[i]; r[i] -= bass[i]; }
    const auto rec = hpf::recommend({l, r}, SR);
    CHECK(rec.keep_hz && cents(*rec.keep_hz, E1) < 50);
    CHECK(rec.cutoff_hz && hpf::design::attenuation_db(E1, *rec.cutoff_hz, rec.order) <= 1.02);
}

TEST(band_dof_known_cases) {
    NEAR(hpf::analysis::band_dof(5, 0.0, 5, 1.0), 2.0, 1e-12);
    NEAR(hpf::analysis::band_dof(5, 0.5, 6, 0.5), 2.0 / (0.5 + 2 * (4.0 / 9.0) * 0.25), 1e-12);
    NEAR(hpf::analysis::band_dof(10, 0.0, 410, 0.0) / 400.0, 36.0 / 35.0, 0.01);
}

TEST(json_string_escapes) {
    CHECK(hpf::json_string("a\"b\\c.wav") == "\"a\\\"b\\\\c.wav\"");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    int ran = 0, failed_tests = 0;
    for (auto& [name, fn] : tests()) {
        if (!only.empty() && name.find(only) == std::string::npos) continue;
        g_current = name;
        const int before = g_failed;
        fn();
        ++ran;
        const bool ok = g_failed == before;
        failed_tests += ok ? 0 : 1;
        std::printf("%s %s\n", ok ? "ok   " : "FAIL ", name.c_str());
    }
    std::printf("\n%d tests, %d checks, %d failed tests\n", ran, g_checks, failed_tests);
    return failed_tests ? 1 : 0;
}
