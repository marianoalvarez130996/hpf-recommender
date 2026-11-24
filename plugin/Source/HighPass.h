// Pasa-altos Butterworth de orden 1 a 8 (6 a 48 dB/oct) para el hilo de audio.
//
// El diseño no reserva memoria, así que los coeficientes se pueden recalcular dentro de processBlock
// mientras el corte se suaviza. Es el mismo filtro que hpf::design::sos (transformación bilineal con
// prewarping, como scipy.signal.butter) y que reporta hpf::recommend; la diferencia es que cada
// sección se normaliza a ganancia 1 en Nyquist en lugar de poner toda la ganancia en la primera.
// Las secciones van en forma directa transpuesta II, como hpf::apply_hpf, pero el estado se conserva
// entre bloques.
#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

namespace hpfplugin {

struct Biquad {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
};

inline constexpr int maxOrder = 8;

struct Coefficients {
    std::array<Biquad, (maxOrder + 1) / 2> section{};
    int sections = 0;
};

// Órdenes pares: order/2 secciones de segundo orden; impares: una de primer orden y (order-1)/2 de
// segundo (orden 1: solo la de primer orden).
inline Coefficients butterworthHighPass(double fc, int order, double sampleRate) noexcept
{
    constexpr double pi = 3.14159265358979323846;
    const double k = 2.0 * sampleRate;
    const double wc = k * std::tan(pi * fc / sampleRate);       // corte analógico con prewarping
    Coefficients c;
    for (int m = 1 - order % 2; m < order; m += 2) {          // m = 0 solo en órdenes impares
        const std::complex<double> prototype = -std::polar(1.0, pi * m / (2.0 * order));
        const std::complex<double> s = wc / prototype;           // pasa-bajos prototipo -> pasa-altos
        const std::complex<double> z = (k + s) / (k - s);        // bilineal
        Biquad& b = c.section[static_cast<std::size_t>(c.sections++)];
        if (m == 0) {                                             // polo real de los órdenes impares
            const double g = (1.0 + z.real()) / 2.0;
            b = {g, -g, 0.0, -z.real(), 0.0};
        } else {                                                  // par conjugado
            const double a1 = -2.0 * z.real(), a2 = std::norm(z);
            const double g = (1.0 - a1 + a2) / 4.0;
            b = {g, -2.0 * g, g, a1, a2};
        }
    }
    return c;
}

// Cascada con estado por canal (mono o estéreo).
class HighPass {
public:
    static constexpr int maxChannels = 2;

    void reset() noexcept
    {
        for (auto& channel : state)
            for (auto& s : channel) s = {};
    }

    void setCoefficients(const Coefficients& c) noexcept { coefficients = c; }

    void process(float* const* data, int numChannels, int numSamples) noexcept
    {
        for (int ch = 0; ch < numChannels && ch < maxChannels; ++ch) {
            float* x = data[ch];
            for (int i = 0; i < coefficients.sections; ++i) {
                const Biquad& b = coefficients.section[static_cast<std::size_t>(i)];
                State& s = state[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)];
                for (int n = 0; n < numSamples; ++n) {
                    const double in = x[n];
                    const double out = b.b0 * in + s.z1;
                    s.z1 = b.b1 * in - b.a1 * out + s.z2;
                    s.z2 = b.b2 * in - b.a2 * out;
                    x[n] = static_cast<float>(out);
                }
            }
        }
    }

private:
    struct State {
        double z1 = 0.0, z2 = 0.0;
    };
    Coefficients coefficients;
    std::array<std::array<State, (maxOrder + 1) / 2>, maxChannels> state{};
};

}  // namespace hpfplugin
