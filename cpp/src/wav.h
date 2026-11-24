#pragma once

#include <cstdint>
#include <string>

#include "hpf/hpf.h"

namespace hpf::wav {

struct Format {
    uint16_t tag = 1;          // 1 = PCM, 3 = IEEE float
    uint16_t bits = 24;
    uint16_t channels = 1;
    uint32_t sample_rate = 48000;
};

// Lee WAV PCM 8/16/24/32 bit o float 32/64 (también WAVE_FORMAT_EXTENSIBLE). Las muestras
// enteras se escalan por 2^(bits-1), como libsndfile.
Channels read(const std::string& path, Format& fmt);

// Escribe en el mismo formato; en PCM recorta a [-1, 1). Devuelve el pico antes de recortar.
double write(const std::string& path, const Channels& x, const Format& fmt);

}  // namespace hpf::wav
