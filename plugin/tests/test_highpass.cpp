// Comprueba el filtro del plugin contra el núcleo: la misma respuesta en frecuencia que
// hpf::design::sos y la misma salida que hpf::apply_hpf aunque se procese en bloques.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include <hpf/hpf.h>

#include "HighPass.h"
#include "design.h"
#include "dsp.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
int failures = 0;

void check(bool ok, const char* what, double value)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s (%g)\n", what, value);
    }
}

std::complex<double> response(const hpfplugin::Coefficients& c, double f, double sr)
{
    const std::complex<double> z1 = std::polar(1.0, -2.0 * kPi * f / sr), z2 = z1 * z1;
    std::complex<double> h = 1.0;
    for (int i = 0; i < c.sections; ++i) {
        const auto& b = c.section[static_cast<std::size_t>(i)];
        h *= (b.b0 + b.b1 * z1 + b.b2 * z2) / (1.0 + b.a1 * z1 + b.a2 * z2);
    }
    return h;
}

}  // namespace

int main()
{
    double worstDb = 0.0, worstDeg = 0.0;
    for (double sr : {44100.0, 48000.0, 96000.0})
        for (int order = 1; order <= hpfplugin::maxOrder; ++order)
            for (double fc : {16.0, 20.0, 28.0, 31.5, 56.0, 90.0, 140.0, 300.0}) {
                const auto plugin = hpfplugin::butterworthHighPass(fc, order, sr);
                const auto core = hpf::design::sos(fc, order, sr);
                for (double f : {5.0, 10.0, 20.0, 41.2, 82.4, 100.0, 250.0, 1000.0, 10000.0}) {
                    const auto a = response(plugin, f, sr), b = hpf::dsp::sosfreqz(core, f, sr);
                    worstDb = std::max(worstDb, std::abs(20.0 * std::log10(std::abs(a) / std::abs(b))));
                    worstDeg = std::max(worstDeg, std::abs(std::arg(a / b)) * 180.0 / kPi);
                }
            }
    // la mayor diferencia (unos 2e-8 dB) aparece a 5 Hz con 96 kHz: ahí z está tan cerca de 1 que
    // el numerador (1 - 1/z)^2 se cancela y el redondeo de los dos diseños difiere en ~1e-9 relativo
    check(worstDb < 1e-6, "magnitude differs from hpf::design::sos, dB", worstDb);
    check(worstDeg < 1e-9, "phase differs from hpf::design::sos, degrees", worstDeg);

    // bloques de tamaño aleatorio contra apply_hpf sobre la señal completa
    std::mt19937 rng(7);
    std::normal_distribution<double> noise(0.0, 0.1);
    std::uniform_int_distribution<int> block(1, 1024);
    const double sr = 48000.0;
    const int n = 48000;
    double worstSample = 0.0;
    for (int order = 1; order <= hpfplugin::maxOrder; ++order) {
        hpf::Channels reference(2, std::vector<double>(static_cast<std::size_t>(n)));
        std::vector<std::vector<float>> io(2, std::vector<float>(static_cast<std::size_t>(n)));
        for (std::size_t c = 0; c < 2; ++c)
            for (std::size_t i = 0; i < io[c].size(); ++i) {
                io[c][i] = static_cast<float>(noise(rng));
                reference[c][i] = io[c][i];
            }
        hpf::apply_hpf(reference, sr, 56.0, order);
        hpfplugin::HighPass hp;
        hp.setCoefficients(hpfplugin::butterworthHighPass(56.0, order, sr));
        for (int start = 0; start < n;) {
            const int len = std::min(block(rng), n - start);
            float* channels[2] = {io[0].data() + start, io[1].data() + start};
            hp.process(channels, 2, len);
            start += len;
        }
        for (std::size_t c = 0; c < 2; ++c)
            for (std::size_t i = 0; i < io[c].size(); ++i)
                worstSample = std::max(worstSample, std::abs(io[c][i] - reference[c][i]));
    }
    check(worstSample < 1e-6, "block output differs from hpf::apply_hpf", worstSample);

    std::printf("%s. Difference from the core: magnitude %.1e dB, phase %.1e degrees, sample %.1e\n",
                failures ? "With failures" : "Passed", worstDb, worstDeg, worstSample);
    return failures ? 1 : 0;
}
