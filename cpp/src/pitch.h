#pragma once

#include <optional>
#include <vector>

#include "hpf/hpf.h"

namespace hpf::pitch {

constexpr int kAnalysisSr = 16000;   // para f0 <= 1 kHz sobra; YIN cuesta ~3 veces menos que a 48 kHz

struct YinResult {
    std::vector<double> f0;      // NaN sin periodo claro
    std::vector<double> power;
    std::vector<long> centers;
    std::vector<double> aper;    // d' en el periodo elegido
    std::vector<double> aper2;   // d' en el periodo doble
};

YinResult yin(const std::vector<double>& x, double sr, double fmin, double fmax, double hop_s = 0.010,
              double threshold = 0.15);

std::vector<double> verify_octave(const std::vector<double>& x, double sr, const std::vector<double>& f0,
                                  const std::vector<long>& centers, double fmin, const std::vector<double>* aper,
                                  const std::vector<double>* aper2, int n_harm = 8, double prom_db = 8.0,
                                  double drop_db = 25.0, std::size_t nfft = 4096);

std::optional<double> lowest_note(const std::vector<double>& f0, int min_frames = 12, int min_run = 6,
                                  double width_cents = 100.0);

constexpr double kFrameSnrDb = 10.0;   // en una banda dominada por el piso, lo que una nota sobresale

struct Steady {
    const std::vector<double>* centers;
    std::vector<bool> mask;                              // piso > media - 3 dB
    const std::vector<std::vector<double>>* frame_db;   // [trama corta][banda]
    const std::vector<double>* floor_db;
    double t0 = 0.0;                                     // centro de la primera trama corta (s)
    double hop = 0.0;                                    // salto entre tramas cortas (s)
};

struct PitchResult {
    std::optional<double> f0_min;
    double voiced_energy = 0.0;   // parte de la energía activa en tramas con nota verificada
    long n_voiced = 0;
};

PitchResult estimate_min_f0(const Channels& x, double sr, double fmin = 27.5, double fmax = 1000.0,
                            double gate_db = 35.0, const std::vector<double>& exclude_hz = {},
                            const Steady* steady = nullptr);

}  // namespace hpf::pitch
