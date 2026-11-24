// hpf_recommend archivo.wav [...] [--export] [--json salida.json] [--order N] ...
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/wav.h"
#include "hpf/hpf.h"

namespace {

void usage() {
    std::cerr << "uso: hpf_recommend archivo.wav [...] [--export] [--json r.json] [--max_loss dB]\n"
                 "     [--edge_loss dB] [--range_db dB] [--fmin Hz] [--fmax Hz] [--order N]\n"
                 "     [--max_order N] [--fc_max Hz] [--no_snap]\n";
}

std::string stem_of(const std::string& path, std::string& dir, std::string& ext) {
    const auto slash = path.find_last_of('/');
    dir = slash == std::string::npos ? "" : path.substr(0, slash + 1);
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const auto dot = name.find_last_of('.');
    ext = dot == std::string::npos ? "" : name.substr(dot);
    return dot == std::string::npos ? name : name.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
    hpf::Params p;
    std::vector<std::string> inputs;
    std::string json_path;
    bool do_export = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> double {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return std::atof(argv[++i]);
        };
        if (a == "--export") do_export = true;
        else if (a == "--json") { if (i + 1 >= argc) { usage(); return 2; } json_path = argv[++i]; }
        else if (a == "--max_loss") p.max_loss_db = next();
        else if (a == "--edge_loss") p.edge_loss_db = next();
        else if (a == "--range_db") p.range_db = next();
        else if (a == "--fmin") p.fmin = next();
        else if (a == "--fmax") p.fmax = next();
        else if (a == "--order") p.order = int(next());
        else if (a == "--max_order") p.max_order = int(next());
        else if (a == "--fc_max") p.fc_max = next();
        else if (a == "--no_snap") p.snap = false;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') { usage(); return 2; }
        else inputs.push_back(a);
    }
    if (inputs.empty()) { usage(); return 2; }

    std::vector<std::string> reports;
    int status = 0;
    for (const auto& path : inputs) {
        try {
            hpf::wav::Format fmt;
            auto x = hpf::wav::read(path, fmt);
            const auto r = hpf::recommend(x, fmt.sample_rate, p);
            std::string dir, ext;
            const std::string stem = stem_of(path, dir, ext);
            std::string body = hpf::to_json(r, 2);
            std::string head = "{\n  \"file\": " + hpf::json_string(stem + ext) + ",\n  \"sr\": " +
                               std::to_string(fmt.sample_rate) + ",\n  \"channels\": " + std::to_string(fmt.channels) + ",\n";
            // sin nada medible que quitar se exporta solo el filtro fijo
            const bool chain = r.active && r.cutoff_hz;
            if (do_export && (chain || r.base_hz)) {
                hpf::apply_recommendation(x, fmt.sample_rate, r);
                char fc[32];
                std::snprintf(fc, sizeof fc, "%g", chain ? *r.cutoff_hz : *r.base_hz);
                const std::string out = dir + stem + "_HPF_" + fc + "Hz_ord" +
                                        std::to_string(chain ? r.order : r.base_order) + ext;
                const double peak = hpf::wav::write(out, x, fmt);
                if (peak > 1.0 && fmt.tag == 1)
                    std::fprintf(stderr, "Aviso: pico de %+.2f dBFS en %s; el PCM va a recortar.\n",
                                 20 * std::log10(peak), path.c_str());
                body.insert(body.size() - 2, ",\n  \"exported\": " + hpf::json_string(out));
            }
            reports.push_back(head + body.substr(2));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
            status = 1;
        }
    }
    std::string text;
    if (reports.size() == 1) text = reports[0];
    else {
        text = "[\n";
        for (std::size_t i = 0; i < reports.size(); ++i) text += reports[i] + (i + 1 < reports.size() ? ",\n" : "\n");
        text += "]";
    }
    std::cout << text << "\n";
    if (!json_path.empty()) std::ofstream(json_path) << text << "\n";
    return status;
}
