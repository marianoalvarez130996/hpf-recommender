#include "dsp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace hpf::dsp {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

// Radix 4 con entrada en orden de bits invertido (decimación en el tiempo). Cada etapa junta
// dos etapas radix 2: con w = W_4h^j, los cuatro puntos a0..a3 salen con tres productos
// (a1·w², a2·w, a3·w³) en lugar de cuatro, y el arreglo se recorre la mitad de veces.
Fft::Fft(std::size_t n) : n_(n), odd_(false) {
    if (n == 0 || (n & (n - 1)) != 0)
        throw std::invalid_argument("FFT: the size must be a power of 2");
    std::size_t bits = 0;
    while ((std::size_t{1} << bits) < n) ++bits;
    odd_ = (bits % 2) == 1;
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t r = 0;
        for (std::size_t b = 0; b < bits; ++b)
            if (i & (std::size_t{1} << b)) r |= std::size_t{1} << (bits - 1 - b);
        if (i < r) { rev_.push_back(i); rev_.push_back(r); }
    }
    for (std::size_t h = odd_ ? 2 : 1; h < n; h *= 4)
        for (std::size_t j = 0; j < h; ++j)
            for (int m = 1; m <= 3; ++m) {
                const double ang = -2.0 * kPi * double(m) * double(j) / double(4 * h);
                tw_.push_back(std::cos(ang));
                tw_.push_back(std::sin(ang));
            }
}

// aritmética compleja a mano: std::complex::operator* sigue el anexo G de C99 y es lento
template <bool Inv>
void Fft::run(cplx* a) const {
    for (std::size_t k = 0; k < rev_.size(); k += 2) std::swap(a[rev_[k]], a[rev_[k + 1]]);
    double* d = reinterpret_cast<double*>(a);
    const double sg = Inv ? -1.0 : 1.0;   // conjuga los factores de giro en la inversa
    std::size_t h = 1;
    if (odd_) {
        for (std::size_t i = 0; i < 2 * n_; i += 4) {
            const double ur = d[i], ui = d[i + 1], vr = d[i + 2], vi = d[i + 3];
            d[i] = ur + vr; d[i + 1] = ui + vi;
            d[i + 2] = ur - vr; d[i + 3] = ui - vi;
        }
        h = 2;
    }
    const double* tw = tw_.data();
    for (; h < n_; h *= 4) {
        for (std::size_t base = 0; base < n_; base += 4 * h) {
            double* p0 = d + 2 * base;
            double* p1 = p0 + 2 * h;
            double* p2 = p1 + 2 * h;
            double* p3 = p2 + 2 * h;
            for (std::size_t j = 0; j < h; ++j) {
                const double* w = tw + 6 * j;
                const double w1r = w[0], w1i = sg * w[1];
                const double w2r = w[2], w2i = sg * w[3];
                const double w3r = w[4], w3i = sg * w[5];
                const std::size_t q = 2 * j;
                const double a0r = p0[q], a0i = p0[q + 1];
                const double b1r = p1[q] * w2r - p1[q + 1] * w2i, b1i = p1[q] * w2i + p1[q + 1] * w2r;
                const double b2r = p2[q] * w1r - p2[q + 1] * w1i, b2i = p2[q] * w1i + p2[q + 1] * w1r;
                const double b3r = p3[q] * w3r - p3[q + 1] * w3i, b3i = p3[q] * w3i + p3[q + 1] * w3r;
                const double s0r = a0r + b1r, s0i = a0i + b1i, d0r = a0r - b1r, d0i = a0i - b1i;
                const double s1r = b2r + b3r, s1i = b2i + b3i, d1r = b2r - b3r, d1i = b2i - b3i;
                // ∓i·d1: -i en la directa, +i en la inversa
                const double jr = sg * d1i, ji = -sg * d1r;
                p0[q] = s0r + s1r; p0[q + 1] = s0i + s1i;
                p2[q] = s0r - s1r; p2[q + 1] = s0i - s1i;
                p1[q] = d0r + jr;  p1[q + 1] = d0i + ji;
                p3[q] = d0r - jr;  p3[q + 1] = d0i - ji;
            }
        }
        tw += 6 * h;
    }
    if (Inv) {
        const double s = 1.0 / double(n_);
        for (std::size_t i = 0; i < 2 * n_; ++i) d[i] *= s;
    }
}

void Fft::forward(cplx* a) const { run<false>(a); }
void Fft::inverse(cplx* a) const { run<true>(a); }

RealFft::RealFft(std::size_t n) : n_(n), half_(n / 2), w_(n / 2), buf_(n / 2) {
    if (n < 4) throw std::invalid_argument("RealFft: n >= 4");
    for (std::size_t k = 0; k < n / 2; ++k)
        w_[k] = std::polar(1.0, -2.0 * kPi * double(k) / double(n));
}

// Los bins k y h-k salen de los mismos dos valores de Z, así que el ciclo los calcula juntos.
// Escrito así, GCC tampoco intenta vectorizar el acceso en reversa (con -O3 eso lo hacía 7 veces
// más lento).
void RealFft::forward(const double* x, std::size_t len, cplx* out) const {
    const std::size_t h = n_ / 2;
    len = std::min(len, n_);
    for (std::size_t m = 0; m < h; ++m) {
        const double re = 2 * m < len ? x[2 * m] : 0.0;
        const double im = 2 * m + 1 < len ? x[2 * m + 1] : 0.0;
        buf_[m] = cplx(re, im);
    }
    half_.forward(buf_.data());
    // Z = FFT(x_par + i·x_impar): E = (Z[k] + conj Z[h-k])/2, O = (Z[k] - conj Z[h-k])/2i,
    // X[k] = E + W^k·O y X[h-k] = conj(E - W^k·O)
    const double* z = reinterpret_cast<const double*>(buf_.data());
    const double* w = reinterpret_cast<const double*>(w_.data());
    double* o = reinterpret_cast<double*>(out);
    o[0] = z[0] + z[1]; o[1] = 0.0;
    o[2 * h] = z[0] - z[1]; o[2 * h + 1] = 0.0;
    for (std::size_t k = 1; k <= h / 2; ++k) {
        const std::size_t m = h - k;
        const double ar = z[2 * k], ai = z[2 * k + 1], br = z[2 * m], bi = z[2 * m + 1];
        const double er = 0.5 * (ar + br), ei = 0.5 * (ai - bi);
        const double orr = 0.5 * (ai + bi), oi = -0.5 * (ar - br);
        const double wr = w[2 * k], wi = w[2 * k + 1];
        const double tr = wr * orr - wi * oi, ti = wr * oi + wi * orr;
        o[2 * k] = er + tr;      o[2 * k + 1] = ei + ti;
        o[2 * m] = er - tr;      o[2 * m + 1] = ti - ei;
    }
}

// Inverso del paso anterior: E = (X[k] + conj X[h-k])/2, O = conj(W^k)·(X[k] - conj X[h-k])/2,
// Z[k] = E + i·O y Z[h-k] = conj(E) + i·conj(O)
void RealFft::inverse(const cplx* spec, double* out) const {
    const std::size_t h = n_ / 2;
    const double* x = reinterpret_cast<const double*>(spec);
    const double* w = reinterpret_cast<const double*>(w_.data());
    double* z = reinterpret_cast<double*>(buf_.data());
    {
        const double er = 0.5 * (x[0] + x[2 * h]), ei = 0.5 * (x[1] - x[2 * h + 1]);
        const double orr = 0.5 * (x[0] - x[2 * h]), oi = 0.5 * (x[1] + x[2 * h + 1]);
        z[0] = er - oi; z[1] = ei + orr;
    }
    for (std::size_t k = 1; k <= h / 2; ++k) {
        const std::size_t m = h - k;
        const double ar = x[2 * k], ai = x[2 * k + 1], br = x[2 * m], bi = x[2 * m + 1];
        const double er = 0.5 * (ar + br), ei = 0.5 * (ai - bi);
        const double dr = 0.5 * (ar - br), di = 0.5 * (ai + bi);
        const double wr = w[2 * k], wi = -w[2 * k + 1];
        const double orr = dr * wr - di * wi, oi = dr * wi + di * wr;
        z[2 * k] = er - oi;      z[2 * k + 1] = ei + orr;
        z[2 * m] = er + oi;      z[2 * m + 1] = orr - ei;
    }
    half_.inverse(buf_.data());
    for (std::size_t m = 0; m < h; ++m) {
        out[2 * m] = buf_[m].real();
        out[2 * m + 1] = buf_[m].imag();
    }
}

std::size_t next_pow2(double v) {
    return std::size_t{1} << static_cast<int>(std::ceil(std::log2(v)));
}

std::vector<double> hann_periodic(std::size_t n) {
    std::vector<double> w(n);
    for (std::size_t i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * double(i) / double(n));
    return w;
}

namespace {

double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double q = x * x / 4.0;
    for (int k = 1; k < 500; ++k) {
        term *= q / (double(k) * double(k));
        sum += term;
        if (term < sum * 1e-17) break;
    }
    return sum;
}

double sinc(double x) {
    if (x == 0.0) return 1.0;
    return std::sin(kPi * x) / (kPi * x);
}

// signal.firwin(numtaps, cutoff, window=("kaiser", 5.0)), pasa-bajos, escalado a ganancia 1 en DC
std::vector<double> firwin_kaiser(long numtaps, double cutoff, double beta) {
    std::vector<double> h(numtaps);
    const double alpha = 0.5 * double(numtaps - 1);
    const double i0b = bessel_i0(beta);
    double s = 0.0;
    for (long n = 0; n < numtaps; ++n) {
        const double m = double(n) - alpha;
        const double r = (double(n) - alpha) / alpha;
        const double win = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;
        h[n] = cutoff * sinc(cutoff * m) * win;
        s += h[n];
    }
    for (auto& v : h) v /= s;
    return h;
}

long output_len(long len_h, long in_len, long up, long down) {
    return (((in_len - 1) * up + len_h) - 1) / down + 1;
}

}  // namespace

std::vector<double> resample_poly(const std::vector<double>& x, long up, long down) {
    const long g = std::gcd(up, down);
    up /= g;
    down /= g;
    if (up == 1 && down == 1) return x;
    const long n_in = static_cast<long>(x.size());
    long n_out = n_in * up;
    n_out = n_out / down + (n_out % down ? 1 : 0);

    const long max_rate = std::max(up, down);
    const long half_len = 10 * max_rate;
    std::vector<double> h = firwin_kaiser(2 * half_len + 1, 1.0 / double(max_rate), 5.0);
    for (auto& v : h) v *= double(up);

    const long n_pre_pad = down - half_len % down;
    long n_post_pad = 0;
    const long n_pre_remove = (half_len + n_pre_pad) / down;
    while (output_len(long(h.size()) + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove)
        ++n_post_pad;
    std::vector<double> hp(n_pre_pad, 0.0);
    hp.insert(hp.end(), h.begin(), h.end());
    hp.insert(hp.end(), n_post_pad, 0.0);
    const long len_h = long(hp.size());

    // upfirdn: y[k] = sum_j hp[j] * u[k*down - j], u = x con up-1 ceros entre muestras
    std::vector<double> y(n_out);
    for (long k = 0; k < n_out; ++k) {
        const long pos = (k + n_pre_remove) * down;
        long j = pos % up;              // j ≡ pos (mod up) para caer en una muestra de x
        double acc = 0.0;
        for (; j < len_h; j += up) {
            const long i = (pos - j) / up;
            if (i < 0) break;
            if (i < n_in) acc += hp[j] * x[i];
        }
        y[k] = acc;
    }
    return y;
}

Sos butter_sos(int order, double fc, double sr, bool highpass) {
    if (order < 1) throw std::invalid_argument("orden >= 1");
    if (!(fc > 0.0 && fc < sr / 2)) throw std::invalid_argument("fc outside (0, sr/2)");
    // Igual que scipy: frecuencias normalizadas con fs = 2, prewarping, bilineal
    const double fs = 2.0;
    const double wn = 2.0 * fc / sr;
    const double warped = 2.0 * fs * std::tan(kPi * wn / fs);
    std::vector<cplx> p;
    for (int m = -order + 1; m < order; m += 2)
        p.push_back(-std::polar(1.0, kPi * double(m) / (2.0 * order)));
    cplx k = 1.0;
    cplx prod_neg_p = 1.0;
    for (auto& v : p) prod_neg_p *= -v;
    if (highpass) {
        k = cplx(std::real(1.0 / prod_neg_p), 0.0);
        for (auto& v : p) v = warped / v;
    } else {
        for (auto& v : p) v *= warped;
        k = std::pow(warped, order);
    }
    // ceros: en s = 0 (pasa-altos) o en el infinito (pasa-bajos)
    const double fs2 = 2.0 * fs;
    cplx num = 1.0, den = 1.0;
    std::vector<cplx> pz;
    for (auto& v : p) {
        den *= fs2 - v;
        pz.push_back((fs2 + v) / (fs2 - v));
    }
    if (highpass) num = std::pow(fs2, order);
    const double gain = std::real(k * num / den);
    const double zero = highpass ? 1.0 : -1.0;   // bilineal de s = 0 -> z = 1; s = inf -> z = -1

    // Pares conjugados (parte imaginaria > 0) y, si el orden es impar, un polo real.
    std::vector<cplx> pairs;
    double real_pole = 0.0;
    bool has_real = false;
    for (auto& v : pz) {
        if (std::abs(v.imag()) < 1e-12) { real_pole = v.real(); has_real = true; }
        else if (v.imag() > 0) pairs.push_back(v);
    }
    // Como zpk2sos: las secciones con polos más cerca del círculo unitario van al final
    std::sort(pairs.begin(), pairs.end(), [](cplx a, cplx b) { return std::abs(a) < std::abs(b); });
    Sos sos;
    if (has_real) sos.push_back({1.0, -zero, 0.0, 1.0, -real_pole, 0.0});
    for (auto& v : pairs)
        sos.push_back({1.0, -2.0 * zero, 1.0, 1.0, -2.0 * v.real(), std::norm(v)});
    for (int i = 0; i < 3; ++i) sos[0][i] *= gain;
    return sos;
}

void sosfilt(const Sos& sos, double* x, std::size_t n) {
    for (const auto& s : sos) {
        double z1 = 0.0, z2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double in = x[i];
            const double out = s[0] * in + z1;
            z1 = s[1] * in - s[4] * out + z2;
            z2 = s[2] * in - s[5] * out;
            x[i] = out;
        }
    }
}

cplx sosfreqz(const Sos& sos, double f, double sr) {
    const cplx e1 = std::polar(1.0, -2.0 * kPi * f / sr);
    const cplx e2 = e1 * e1;
    cplx h = 1.0;
    for (const auto& s : sos)
        h *= (s[0] + s[1] * e1 + s[2] * e2) / (s[3] + s[4] * e1 + s[5] * e2);
    return h;
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const double pos = q / 100.0 * double(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    const double t = pos - double(lo);
    return v[lo] + (v[hi] - v[lo]) * t;
}

double median(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t n = v.size(), mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const double hi = v[mid];
    if (n % 2) return hi;
    const double lo = *std::max_element(v.begin(), v.begin() + mid);
    return 0.5 * (lo + hi);
}

double median_range(const double* begin, const double* end) {
    return median(std::vector<double>(begin, end));
}

namespace {

// P(a, x) regularizada (serie y fracción continua, Numerical Recipes 6.2)
double gamma_p(double a, double x) {
    if (x <= 0.0) return 0.0;
    const double gln = std::lgamma(a);
    if (x < a + 1.0) {
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 0; n < 1000; ++n) {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if (std::abs(del) < std::abs(sum) * 1e-16) break;
        }
        return sum * std::exp(-x + a * std::log(x) - gln);
    }
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -double(i) * (double(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::abs(del - 1.0) < 1e-16) break;
    }
    return 1.0 - std::exp(-x + a * std::log(x) - gln) * h;
}

}  // namespace

double chi2_ppf(double q, double nu) {
    if (!(q > 0.0 && q < 1.0) || !(nu > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    const double a = nu / 2.0;
    // bisección en escala log y luego Newton: P(a, x) es monótona en x
    double lo = 0.0, hi = std::max(1.0, a);
    while (gamma_p(a, hi) < q) hi *= 2.0;
    double x = 0.5 * (lo + hi);
    for (int i = 0; i < 200; ++i) {
        x = 0.5 * (lo + hi);
        if (gamma_p(a, x) < q) lo = x; else hi = x;
        if (hi - lo < 1e-15 * std::max(1.0, x)) break;
    }
    for (int i = 0; i < 5; ++i) {
        const double f = gamma_p(a, x) - q;
        const double dens = std::exp((a - 1.0) * std::log(x) - x - std::lgamma(a));
        if (dens <= 0.0) break;
        const double nx = x - f / dens;
        if (!(nx > lo && nx < hi)) break;
        x = nx;
    }
    return 2.0 * x;
}

}  // namespace hpf::dsp
