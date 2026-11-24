// HPF Butterworth elegido por especificación: pérdida máxima en la nota más grave y
// rechazo mínimo del ruido grave, con el orden más bajo que cumpla ambas.
#include "design.h"

#include <algorithm>
#include <cmath>

namespace hpf::design {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

const std::vector<double>& grid() {
    // pasos de ~12 %, menos que la tolerancia de pérdida: redondear hacia abajo nunca la rompe
    static const std::vector<double> g = {10,  11.2, 12.5, 14,  16,  18,  20,  22.4, 25,  28,  31.5,
                                          35.5, 40,  45,  50,  56,  63,  71,  80,   90,  100, 112,
                                          125, 140, 160, 180, 200, 224, 250, 280,  315, 355, 400,
                                          450, 500, 560, 630, 710, 800, 900, 1000};
    return g;
}

double snap_down(double fc) {
    const auto& g = grid();
    auto it = std::upper_bound(g.begin(), g.end(), fc * (1 + 1e-9));
    const long i = long(it - g.begin()) - 1;
    return g[std::max(0L, i)];
}

double attenuation_db(double f, double fc, int order) {
    return 10.0 * std::log10(1.0 + std::pow(fc / f, 2 * order));
}

double max_cutoff(double f_keep, int order, double max_loss_db) {
    // de |H|² = 1/(1 + (fc/f)^2N) despejando fc para una pérdida max_loss_db en f_keep
    return f_keep * std::pow(std::pow(10.0, max_loss_db / 10.0) - 1.0, 1.0 / (2 * order));
}

Choice choose(double f_keep, const std::vector<double>& junk_hz, const std::vector<double>& junk_need_db,
              double max_loss_db, const std::vector<int>& orders, double fc_min, double fc_max, bool snap,
              double tol_db) {
    Choice last{0.0, 0, true, 0.0};
    for (int n : orders) {
        double fc = std::clamp(max_cutoff(f_keep, n, max_loss_db), fc_min, fc_max);
        if (snap) fc = snap_down(fc);
        double short_db = 0.0;
        if (!junk_hz.empty()) {
            short_db = -1e300;
            for (std::size_t i = 0; i < junk_hz.size(); ++i)
                short_db = std::max(short_db, junk_need_db[i] - attenuation_db(junk_hz[i], fc, n));
        }
        last = Choice{fc, n, short_db <= tol_db, std::max(short_db, 0.0)};
        if (short_db <= tol_db) return last;
    }
    return last;
}

dsp::Sos sos(double fc, int order, double sr) { return dsp::butter_sos(order, fc, sr, true); }

Response response_at(double f, double fc, int order, double sr, double base_hz, int base_order) {
    dsp::Sos s;
    if (base_hz > 0) s = sos(base_hz, base_order, sr);
    const auto rec = sos(fc, order, sr);
    s.insert(s.end(), rec.begin(), rec.end());
    const double df = std::max(0.01, f * 1e-3);
    const double fs[3] = {f - df, f, f + df};
    double ph[3];
    dsp::cplx h1;
    for (int i = 0; i < 3; ++i) {
        const auto h = dsp::sosfreqz(s, fs[i], sr);
        ph[i] = std::arg(h);
        if (i == 1) h1 = h;
    }
    for (int i = 1; i < 3; ++i) {   // np.unwrap
        double d = ph[i] - ph[i - 1];
        while (d > kPi) { ph[i] -= 2 * kPi; d -= 2 * kPi; }
        while (d < -kPi) { ph[i] += 2 * kPi; d += 2 * kPi; }
    }
    const double gd = -(ph[2] - ph[0]) / (2 * kPi * 2 * df);
    return {-20.0 * std::log10(std::abs(h1)), ph[1] * 180.0 / kPi, gd * 1000.0};
}

}  // namespace hpf::design
