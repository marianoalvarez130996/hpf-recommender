#pragma once

#include <optional>
#include <vector>

#include "hpf/hpf.h"

namespace hpf::analysis {

constexpr double kSilenceDb = -100.0;

double to_db(double p);
std::vector<double> band_centers(double f_lo, double f_hi, int per_octave = 24);

// Grados de libertad (Satterthwaite) de una banda que suma bins de Hann con pesos fraccionarios
// en los bordes: i_lo con peso 1 - w_lo, i_hi con peso w_hi, los del medio con peso 1.
double band_dof(std::size_t i_lo, double w_lo, std::size_t i_hi, double w_hi);

struct Spectrum {
    std::vector<double> centers;     // Hz, una banda cada 1/24 oct, de 1/3 oct de ancho
    std::vector<double> mean_db;     // media de las tramas activas
    std::vector<double> floor_db;    // parte estacionaria
    std::vector<double> excess_db;   // media menos piso
    bool stationary = false;
    double active_fraction = 0.0;
    std::vector<double> fine_freqs;  // ~1.5 Hz por bin hasta 500 Hz
    std::vector<double> fine_floor_db;
    std::vector<std::vector<double>> frame_db;   // [trama corta][banda], para la prueba por trama
    double frame_t0 = 0.0;           // centro de la primera trama corta (s)
    double frame_hop = 0.0;          // salto entre tramas cortas (s)

    std::size_t peak_index(double f_lo, double f_hi) const;
};

std::optional<Spectrum> long_term_spectrum(const Channels& x, double sr, double active_range_db = 40.0,
                                           double floor_pct = 10.0, double fine_max_hz = 500.0);

double spectral_edge(const Spectrum& spec, double f_floor, double f_top, double range_db);

struct HumFound {
    double base_hz;
    std::vector<double> harmonics_hz;
    double level_db;
};
std::optional<HumFound> detect_hum(const Spectrum& spec, double prominence_db = 10.0);

}  // namespace hpf::analysis
