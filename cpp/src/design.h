#pragma once

#include <vector>

#include "dsp.h"

namespace hpf::design {

// Serie R20 de números preferentes (ISO 3), de 10 Hz a 1 kHz
const std::vector<double>& grid();
double snap_down(double fc);
double attenuation_db(double f, double fc, int order);   // Butterworth analógico
double max_cutoff(double f_keep, int order, double max_loss_db);

struct Choice {
    double fc;
    int order;
    bool met;
    double short_db;
};
Choice choose(double f_keep, const std::vector<double>& junk_hz, const std::vector<double>& junk_need_db,
              double max_loss_db, const std::vector<int>& orders, double fc_min, double fc_max, bool snap = true,
              double tol_db = 1.0);

dsp::Sos sos(double fc, int order, double sr);

struct Response {
    double loss_db, phase_deg, group_delay_ms;
};
// Con base_hz > 0, el filtro fijo (base_hz, base_order) va delante del recomendado.
Response response_at(double f, double fc, int order, double sr, double base_hz = 0.0, int base_order = 0);

}  // namespace hpf::design
