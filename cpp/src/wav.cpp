#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace hpf::wav {

namespace {

uint32_t u32(const unsigned char* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (uint32_t(p[3]) << 24); }
uint16_t u16(const unsigned char* p) { return uint16_t(p[0] | (p[1] << 8)); }

void put32(std::vector<unsigned char>& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF); }
void put16(std::vector<unsigned char>& b, uint16_t v) { b.push_back(static_cast<unsigned char>(v & 0xFF)); b.push_back(static_cast<unsigned char>(v >> 8)); }

}  // namespace

Channels read(const std::string& path, Format& fmt) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("no se puede abrir " + path);
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.size() < 12 || std::memcmp(data.data(), "RIFF", 4) || std::memcmp(data.data() + 8, "WAVE", 4))
        throw std::runtime_error(path + ": no es un WAV RIFF");
    const unsigned char* pcm = nullptr;
    std::size_t pcm_len = 0;
    bool have_fmt = false;
    for (std::size_t pos = 12; pos + 8 <= data.size();) {
        const unsigned char* ck = data.data() + pos;
        const std::size_t len = u32(ck + 4);
        const std::size_t avail = std::min(len, data.size() - pos - 8);
        if (!std::memcmp(ck, "fmt ", 4)) {
            // se lee lo que el archivo trae, no lo que el bloque declara: uno truncado no puede
            // hacer que se lea más allá del final
            if (avail < 16) throw std::runtime_error(path + ": bloque fmt incompleto");
            fmt.tag = u16(ck + 8);
            fmt.channels = u16(ck + 10);
            fmt.sample_rate = u32(ck + 12);
            fmt.bits = u16(ck + 22);
            if (fmt.tag == 0xFFFE && len >= 40) {   // subformato: primeros 2 bytes del GUID
                if (avail < 26) throw std::runtime_error(path + ": bloque fmt incompleto (EXTENSIBLE)");
                fmt.tag = u16(ck + 8 + 24);
            }
            have_fmt = true;
        } else if (!std::memcmp(ck, "data", 4)) {
            pcm = ck + 8;
            pcm_len = avail;
        }
        pos += 8 + len + (len & 1);
    }
    if (!have_fmt || !pcm) throw std::runtime_error(path + ": falta fmt o data");
    if (fmt.channels == 0) throw std::runtime_error(path + ": cero canales");
    const bool is_float = fmt.tag == 3;
    if (!(fmt.tag == 1 || is_float)) throw std::runtime_error(path + ": formato no soportado");
    if (is_float && fmt.bits != 32 && fmt.bits != 64) throw std::runtime_error(path + ": float de 32 o 64 bit");
    if (!is_float && (fmt.bits < 8 || fmt.bits > 32 || fmt.bits % 8)) throw std::runtime_error(path + ": PCM 8-32 bit");

    const std::size_t bps = fmt.bits / 8, frame = bps * fmt.channels;
    const std::size_t n = pcm_len / frame;
    Channels x(fmt.channels, std::vector<double>(n));
    const double scale = 1.0 / std::ldexp(1.0, fmt.bits - 1);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < fmt.channels; ++c) {
            const unsigned char* s = pcm + i * frame + c * bps;
            double v;
            if (is_float && fmt.bits == 32) { float t; std::memcpy(&t, s, 4); v = t; }
            else if (is_float) { double t; std::memcpy(&t, s, 8); v = t; }
            else if (fmt.bits == 8) v = (double(s[0]) - 128.0) / 128.0;
            else {
                int64_t acc = 0;
                for (std::size_t b = 0; b < bps; ++b) acc |= int64_t(s[b]) << (8 * b);
                const int64_t sign = int64_t(1) << (fmt.bits - 1);
                if (acc & sign) acc -= sign << 1;
                v = double(acc) * scale;
            }
            x[c][i] = v;
        }
    }
    return x;
}

double write(const std::string& path, const Channels& x, const Format& fmt) {
    const std::size_t n = x.empty() ? 0 : x[0].size();
    const std::size_t bps = fmt.bits / 8, ch = x.size();
    std::vector<unsigned char> b;
    b.reserve(44 + n * ch * bps);
    const uint32_t data_len = uint32_t(n * ch * bps);
    b.insert(b.end(), {'R', 'I', 'F', 'F'});
    put32(b, 36 + data_len);
    b.insert(b.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    put32(b, 16);
    put16(b, fmt.tag);
    put16(b, uint16_t(ch));
    put32(b, fmt.sample_rate);
    put32(b, uint32_t(fmt.sample_rate * ch * bps));
    put16(b, uint16_t(ch * bps));
    put16(b, fmt.bits);
    b.insert(b.end(), {'d', 'a', 't', 'a'});
    put32(b, data_len);

    double peak = 0.0;
    const double full = std::ldexp(1.0, fmt.bits - 1);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < ch; ++c) {
            const double v = x[c][i];
            peak = std::max(peak, std::abs(v));
            if (fmt.tag == 3 && fmt.bits == 32) {
                const float t = float(v);
                unsigned char s[4];
                std::memcpy(s, &t, 4);
                b.insert(b.end(), s, s + 4);
            } else if (fmt.tag == 3) {
                unsigned char s[8];
                std::memcpy(s, &v, 8);
                b.insert(b.end(), s, s + 8);
            } else if (fmt.bits == 8) {
                const double q = std::clamp(std::nearbyint(v * 128.0) + 128.0, 0.0, 255.0);
                b.push_back(static_cast<unsigned char>(q));
            } else {
                const double q = std::clamp(std::nearbyint(v * full), -full, full - 1.0);
                const int64_t iv = static_cast<int64_t>(q);
                for (std::size_t k = 0; k < bps; ++k) b.push_back(static_cast<unsigned char>((iv >> (8 * k)) & 0xFF));
            }
        }
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("no se puede escribir " + path);
    f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    return peak;
}

}  // namespace hpf::wav
