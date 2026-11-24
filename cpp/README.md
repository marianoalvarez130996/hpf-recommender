# HPF Recommender 5.2 in C++17

A port of the Python package in `../python`, with the same method, the same defaults, the same
warnings and the same JSON report. It depends on nothing beyond the standard library: the FFT,
polyphase resampling, Butterworth design and WAV reading and writing are in `src/`.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/hpf_tests
```

Version 5.2 builds without warnings (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion
-Wno-sign-conversion`) with Apple Clang 21 and with GCC 16.2 on macOS; with either compiler the tests
pass and the parity check below finds no differences. With Apple Clang,
`-DHPF_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug` builds with AddressSanitizer and
UndefinedBehaviorSanitizer; the tests and the command-line tool run clean under both.

To build with GCC:

```bash
CC=gcc-16 CXX=g++-16 cmake -S . -B build-gcc -DCMAKE_BUILD_TYPE=Release
cmake --build build-gcc
```

## Command line

```bash
./build/hpf_recommend "BASS DI.wav" "GTR 57.wav" --export --json report.json
```

Options match the Python version (`--max_loss`, `--edge_loss`, `--range_db`, `--fmin`, `--fmax`, `--order`,
`--max_order`, `--fc_max`, `--no_snap`, `--export`, `--json`), except `--plot`. It reads PCM WAV
from 8 to 32 bits, 32- and 64-bit float and WAVE_FORMAT_EXTENSIBLE. It does not read RF64, µ-law or
A-law. `--export` writes `<name>_HPF_<fc>Hz_ord<N>.wav`, only when the report recommends filtering, in the same format, channel count and sample
rate, with only the `fmt ` and `data` chunks: `bext`, `iXML` and Pro Tools chunks are not copied.

## Library

```cpp
#include <hpf/hpf.h>

hpf::Channels x = /* one vector<double> per channel */;
hpf::Recommendation r = hpf::recommend(x, 48000.0);
if (r.cutoff_hz) hpf::apply_hpf(x, 48000.0, *r.cutoff_hz, r.order);
std::string json = hpf::to_json(r);
```

`recommend()` analyses a whole track and is not meant for the audio callback. `apply_hpf()` is a
cascade of biquads in transposed direct form II, the same causal filter the report describes.
`recommend()` throws `std::invalid_argument` on NaN or Inf samples, channels of different length,
a non-positive sample rate or parameters out of range.

## Equivalence with Python

Each numerical piece follows the numpy or scipy function it replaces:

| C++ (`src/dsp.cpp`) | Reference |
| --- | --- |
| `RealFft` | `np.fft.rfft` / `np.fft.irfft` |
| `resample_poly` | `scipy.signal.resample_poly` (Kaiser window, β = 5) |
| `butter_sos`, `sosfilt`, `sosfreqz` | `scipy.signal.butter(..., output="sos")`, `sosfilt`, `sosfreqz` |
| `percentile`, `median` | `np.percentile` (linear), `np.median` |
| `chi2_ppf` | `scipy.stats.chi2.ppf` |

Report values are rounded with `nearbyint` (half to even), like Python's `round()`, and numbers are
printed the way Python's `repr()` prints them, so both reports are the same text.

`tests/parity.py` runs both versions on the same files and compares every field: the set of keys,
the discrete fields exactly (cutoff, order, source, note, warnings, hum, parameters) and the
continuous ones within rounding. It runs the fixed cases at 44.1, 48 and 96 kHz and in 24-bit PCM,
120 random cases and any folders of WAV files given on the command line, and checks that the tool
writes valid JSON for file names with quotes or backslashes.

```bash
python tests/parity.py build/hpf_recommend [folder_with_wavs ...]
```

On the 22 tracks of the test session plus the synthetic cases: 194 files, no differences. Two runs
give byte-identical output.

## Tests

`tests/test_main.cpp` (39 tests) covers the real and complex FFT against a direct DFT,
`resample_poly`, `butter_sos` and `chi2_ppf` against scipy values, percentiles and medians, filter
design (loss at the protected frequency, lowest order, R20 series), YIN and the octave check on
signals with a known note, the recommender on the same signals as the Python tests, WAV round trips
in every supported format, malformed WAV headers, the stationary-floor statistics, the JSON output
and invalid parameters.
