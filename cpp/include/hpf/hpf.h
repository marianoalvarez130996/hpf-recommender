// Recomendación de HPF por pista. Port en C++17 de hpf_recommender 5.2 (Python).
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace hpf {

// Audio en memoria: un vector por canal, todos del mismo largo.
using Channels = std::vector<std::vector<double>>;

// Pendientes de 6 a 48 dB/oct, las mismas que ofrece el plugin.
inline constexpr int max_filter_order = 8;
// 24 dB/oct como máximo cuando lo protegido es el borde del espectro.
inline constexpr int edge_max_order = 4;
// Filtro fijo, siempre puesto y antes del análisis: lo que hay bajo 20 Hz no es contenido útil.
inline constexpr double base_filter_hz = 20.0;
inline constexpr int base_filter_order = 4;

struct Params {
    double max_loss_db = 1.0;     // pérdida máxima cuando lo protegido es una nota
    double edge_loss_db = 3.0;    // pérdida en el borde del espectro cuando la fuente no tiene altura
                                  // y el ruido estacionario no llega al borde (si llega, max_loss_db)
    double range_db = 30.0;       // contenido útil: hasta range_db bajo el máximo (bandas de 1/3 oct)
    double fmin = 27.5;           // f0 más grave buscado (A0)
    double fmax = 1000.0;
    double f_floor = 20.0;
    double fc_min = 16.0;
    double fc_max = 300.0;
    int order = 0;                // 1 a 8 (6 a 48 dB/oct); 0 = el más bajo entre 1 y max_order que cumpla
    int max_order = 8;
    double tonal_energy = 0.25;
    bool snap = true;
    double base_hz = base_filter_hz;   // filtro fijo de orden base_filter_order delante del recomendado;
                                       // 0 = sin filtro fijo
};

struct Hum {
    double base_hz = 0.0;
    std::vector<double> harmonics_hz;
    double level_db_re_peak = 0.0;
};

struct Recommendation {
    std::optional<double> cutoff_hz;
    int order = 0;
    int slope_db_oct = 0;
    bool active = false;              // si conviene filtrar: hay algo medible que quitar
    std::optional<double> keep_hz;
    std::string source = "ninguna";   // f0 | espectro | mixto | ninguna
    std::optional<double> loss_at_keep_db;
    std::optional<double> loss_budget_db;   // pérdida permitida en keep_hz
    std::optional<double> base_hz;          // filtro fijo, siempre puesto, delante del recomendado
    int base_order = 0;
    std::optional<double> phase_at_keep_deg;
    std::optional<double> group_delay_ms;
    bool rejection_met = true;
    double rejection_short_db = 0.0;
    std::optional<double> f0_min_hz;
    std::optional<std::string> f0_note;
    double voiced_energy = 0.0;
    std::optional<double> spectral_edge_hz;
    std::optional<double> peak_band_hz;
    std::optional<Hum> hum;
    std::vector<double> dc_offset_dbfs;
    std::vector<std::string> warnings;
    Params params;                      // los que se usaron, como "params" en el reporte de Python
};

// Lanza std::invalid_argument si hay NaN/Inf, canales de distinto largo, sr <= 0 o parámetros
// fuera de rango (fmin, fmax, fc_min, fc_max, max_loss_db, range_db, órdenes).
Recommendation recommend(const Channels& x, double sr, const Params& p = Params{});

// HPF Butterworth causal, en el lugar. Mismo filtro que reporta recommend().
void apply_hpf(Channels& x, double sr, double fc, int order);
// La cadena completa de una recomendación: el filtro fijo y, si active, el recomendado.
void apply_recommendation(Channels& x, double sr, const Recommendation& r);

std::string to_json(const Recommendation& r, int indent = 2);
std::string json_string(const std::string& s);   // cadena JSON con comillas y escapes
std::string note_name(double f);

}  // namespace hpf
