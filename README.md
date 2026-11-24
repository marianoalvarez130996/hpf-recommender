# HPF Recommender

Automatic high-pass filter recommendation for the tracks of a multitrack session. For each track it
decides every parameter of the filter (whether to filter at all, the corner frequency and the slope,
from 6 to 48 dB/oct) and writes a short report of why: the frequency it protected and where that
frequency came from, what the filter costs there, and what it could not decide.

**Author:** Mariano Alvarez

## Why

Setting a high-pass filter on every track is part of the maintenance that comes before the creative
part of a mix. The recommender does that part in a few seconds per track and leaves a report, so the
engineer reviews the proposals instead of setting each filter by ear, and spends the attention saved
on balance, space and character. It was first written as the pre-filter stage of a masking analysis:
energy below an instrument's useful range counts as overlap between tracks unless it is removed first.

## What it does

0. A fixed 20 Hz high-pass at 24 dB/oct is always on; everything below works on its output, and the
   recommended filter comes after it, sharing the loss budget.
1. Long-term spectrum in one-third-octave bands, with a stationary noise floor (10th percentile over
   time, bias-corrected with the band's degrees of freedom) subtracted from it.
2. Lowest note: YIN pitch estimates checked against each frame's spectrum; frames that sit on mains
   hum or on a floor-dominated band at that moment are discarded.
3. Protected frequency: the lowest note when pitched frames carry at least a quarter of the energy,
   otherwise the lower edge of the spectrum.
4. Loss allowed there: 1 dB at a note. At the edge of an unpitched source, 3 dB (the corner sits on
   the edge, which is already 30 dB under the maximum), unless stationary noise reaches the edge
   (a floor-dominated band less than a third of an octave below it): subtracting the floor then
   also removes part of the source's skirt, the measured edge is only an upper bound, and 1 dB is
   used.
5. Whether to filter: only if some band below the transition region rises above the usefulness
   threshold (30 dB under the spectral maximum) or there is a DC offset. Otherwise the report
   recommends leaving the filter off, because it would remove nothing measurable.
6. Slope and corner: the gentlest Butterworth slope, from 6 to 48 dB/oct (orders 1 to 8), whose
   highest corner within the allowed loss pushes that noise down to the threshold. When the
   protected frequency is a spectral edge the slope stops at 24 dB/oct: the edge is measured with
   error, and a steeper filter turns the same error into more loss. The corner is rounded down to
   the R20 series of preferred numbers. The curve is always Butterworth, the flattest one: it
   neither boosts nor ripples above the corner.

The report also gives the phase and group delay at the protected frequency, which matter when two
microphones on the same source get different filters, and flags mains hum that a high-pass filter
cannot remove.

## Use

Python with numpy, scipy and soundfile (matplotlib for `--plot`). Tested with Python 3.14, numpy
2.5.4, scipy 1.18.1 and soundfile 0.14.0:

```bash
cd python
pip install -r requirements.txt
python -m hpf_recommender "BASS DI.wav" "GTR 57.wav" --export --json report.json
python -m pytest tests
```

```python
import soundfile as sf
from hpf_recommender import recommend, apply_hpf

x, sr = sf.read("bass.wav", always_2d=True)
r = recommend(x, sr)
y = apply_hpf(x, sr, r.cutoff_hz, r.order)
```

C++17, no dependencies beyond the standard library (see `cpp/README.md`):

```bash
cmake -S cpp -B cpp/build && cmake --build cpp/build && ./cpp/build/hpf_tests
./cpp/build/hpf_recommend "BASS DI.wav" --export
python cpp/tests/parity.py cpp/build/hpf_recommend
```

Always analyse the original file, not one that has already been filtered. Code comments are in
Spanish.

## Repository

| Path | Contents |
| --- | --- |
| `python/` | The `hpf_recommender` package (version 5.2) and its tests |
| `cpp/` | The C++17 port with the same report, its tests and the Python/C++ parity check |
| `evaluation/` | Scripts behind every reported number, and their results in `evaluation/results/` |
| `plugin/` | **HPF!**, the AU/VST3 plugin by Atonal built on `cpp/` (JUCE 9), with its own tests and licence |

The built plugin (VST3 for macOS, Windows and Linux, Audio Unit for macOS, and a standalone application)
is attached to the repository's releases, built and tested by `.github/workflows/build.yml`; see
`plugin/INSTALL.md`.

## Reproducing the evaluation

From `evaluation/`, with the Python requirements installed:

```bash
python fixed_cases.py
for s in 2026 $(seq 1 20) $(seq 31 40); do python battery.py --seed $s; python baselines.py --seed $s; python ablation_slope.py --seed $s; done
python real_tracks.py "path/to/session"
python session_energy.py "path/to/session"
python cross_track.py "path/to/session"
python floor_calibration.py
python limits.py
python real_annotated.py idmt path/to/IDMT-SMT-BASS-SINGLE-TRACKS
python real_annotated.py guitarset path/to/GuitarSet
python unannotated_tone.py path/to/IDMT-SMT-BASS-SINGLE-TRACKS
python off_cases.py 31 40
python plugin_host.py path/to/hpf_host_test     # built from plugin/, see plugin/README.md
python summarize.py
```

`battery.py` draws pitched lines with a known lowest note and unpitched signals, with narrowband
rumble and, in 30 % of the cases, 60 Hz hum, and analyses each one with and without the frame test of
the pitch stage. `baselines.py` scores the cut of every method on the same signals: the recommender,
fixed corners at 40, 80 and 120 Hz, the 100 Hz rule of a knowledge-engineered mixing system, a corner
20 % below a plain YIN estimate, the corner on the spectral edge alone, and two ablations of the edge
budget; `ablation_slope.py` scores slopes up to 48 dB/oct at spectral edges. A cut protects when its
loss at the reference frequency stays within budget (1 dB at a note, 3 dB at the clean source's edge)
and is exact when it also lies within one R20 step of the highest corner that protects.

Seed 2026 was used while developing the method and seeds 1 to 20 while revising it, including the
edge budget, the slope limit and the fixed 20 Hz stage; seeds 31 to 40 were generated once, with the
code frozen (`results/code_frozen_sha256.txt`), and are the test set (`python off_cases.py 31 40` reads
which test lines are left to the fixed stage). `real_annotated.py` scores the same methods
on recordings with annotated notes: the 360 microphone takes of GuitarSet (Zenodo, CC BY 4.0) and the
17 bass lines of IDMT-SMT-Bass-Single-Track (Zenodo), as recorded and with added rumble and hum.
Neither dataset nor the multitrack session used in the paper is included; `results/tracks/` holds the
per-track reports and spectra derived from the session.

`evaluation/listening/` holds a blind ABX test for one listener: `prepare_abx.py` cuts excerpts of ten
distinct tracks of a session (unfiltered, with the corner the recommender computes, and with that
corner one octave higher) into a folder outside the repository, 20 comparisons in all, and `abx.html`,
opened in a browser, plays them and saves the answers. The author's answers (one trial per comparison)
are in `evaluation/results/listening_author.json`.

## Results in brief

Every track first passes a fixed 20 Hz high-pass at 24 dB/oct; the analysis runs on its output and the
recommended filter comes after it, sharing the loss budget. On the test seeds (31 to 40, generated after
the code was frozen), the recommender recovers the lowest note within 50 cents in 99.45 % of 2000 pitched
lines (Wilson 95 % interval 99.0 to 99.7 %). Its cut keeps that note within the 1 dB budget in 99.70 % of
the lines and lies within one R20 step of the highest corner that does in 92.25 % (99.2 % of the lines it
filters; the 7.05 % it leaves to the fixed stage count as not exact); the 100 Hz rule protects 66.55 % and
a corner 20 % below a plain YIN estimate is exact in 92.90 %. On 1000 unpitched signals it protects the
clean source's edge in 93.9 % of cases and is exact in 33.9 %. On recordings with annotated notes it
protects the lowest note in 16 of 17 bass lines; on 360 acoustic-guitar takes it protects 97.8 % as
recorded and 81.1 % with added rumble, because the pitch stage misses short low notes. On a recorded
session it switches the recommended filter on for 4 of 18 distinct signals and leaves the others off,
with nothing measurable to remove; where it filters, the energy above each protected frequency changes
by at most 0.146 dB. `evaluation/summarize.py` gathers
these figures from `evaluation/results/` into `evaluation/results/summary.json`.

## Known limitations

- A single track cannot separate low-frequency bleed from the source's own low end when both have a
  similar level. On drums and room microphones the recommended corners are therefore low.
- Low notes shorter than about 200 ms are not protected. On acoustic guitar played in chords and fast
  lines this is the main source of error, and a fixed resonance of the recording (near 105 Hz in
  GuitarSet) is protected as if it were a note.
- A sine sub-bass has no harmonics to verify, so it is never a note. With another pitched source
  above it, or separated from the main body of the spectrum by a valley, the corner removes it.
- A fundamental more than about 25 dB below the strongest partial is not credible; the octave above
  is protected instead.
- At the edge of an unpitched source the measured edge reads high when stationary noise covers the
  source's skirt, so only about a third of those cuts reach the highest safe corner.
- The only listening check is a blind ABX test by the author, one trial per comparison: 8/10 recommended
  corners and 6/10 controls an octave higher identified (exact one-sided binomial p = 0.055 and 0.377),
  too few trials to show that any single cut is audible. No engineer-labelled corners exist to compare with.

## License

The recommender (`python/`, `cpp/`, `evaluation/`) is under the MIT licence, see `LICENSE`. The plugin
in `plugin/` is under the GNU AGPLv3, see `plugin/LICENSE.md`, as JUCE requires for open-source use.

## Acknowledgements

The recommender began as the pre-filter stage of a spectral unmasking project in development at
Ear Candy Technologies (Mexico City). The rest of that project and the company's code are not included.
