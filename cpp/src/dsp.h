// Piezas de numpy/scipy que usa el recomendador, con la misma semántica.
#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace hpf::dsp {

using cplx = std::complex<double>;

// FFT radix-2 compleja; n potencia de 2. Un plan por tamaño, reutilizable.
class Fft {
public:
    explicit Fft(std::size_t n);
    std::size_t size() const { return n_; }
    void forward(cplx* a) const;
    void inverse(cplx* a) const;   // incluye el 1/n
private:
    template <bool Inv> void run(cplx* a) const;
    std::size_t n_;
    bool odd_;                       // log2(n) impar: una etapa radix 2 antes de las radix 4
    std::vector<std::size_t> rev_;   // pares (i, rev(i)) con i < rev(i)
    std::vector<double> tw_;         // por etapa radix 4 y por j: w, w², w³ (re, im)
};

// FFT real de largo n (potencia de 2) con una compleja de n/2.
class RealFft {
public:
    explicit RealFft(std::size_t n);
    std::size_t size() const { return n_; }
    // x de largo <= n (se rellena con ceros); out: n/2 + 1 bins, como np.fft.rfft(x, n)
    void forward(const double* x, std::size_t len, cplx* out) const;
    // spec: n/2 + 1 bins; out: n muestras, como np.fft.irfft(spec, n)
    void inverse(const cplx* spec, double* out) const;
private:
    std::size_t n_;
    Fft half_;
    std::vector<cplx> w_;          // exp(-2πik/n), k < n/2
    mutable std::vector<cplx> buf_;
};

std::size_t next_pow2(double v);   // 1 << ceil(log2(v))

std::vector<double> hann_periodic(std::size_t n);   // signal.get_window("hann", n)

// signal.resample_poly(x, up, down) con la ventana por defecto (Kaiser, beta 5)
std::vector<double> resample_poly(const std::vector<double>& x, long up, long down);

// Butterworth digital como signal.butter(..., fs=sr, output="sos"); a0 = 1 en cada sección
using Sos = std::vector<std::array<double, 6>>;
Sos butter_sos(int order, double fc, double sr, bool highpass);
void sosfilt(const Sos& sos, double* x, std::size_t n);
cplx sosfreqz(const Sos& sos, double f, double sr);

// np.percentile (lineal) y np.median; v se copia
double percentile(std::vector<double> v, double q);
double median(std::vector<double> v);
double median_range(const double* begin, const double* end);

// stats.chi2.ppf(q, nu)
double chi2_ppf(double q, double nu);

}  // namespace hpf::dsp
