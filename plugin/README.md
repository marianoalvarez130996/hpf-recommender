# HPF! plugin

VST3 (macOS, Windows, Linux), AU (macOS) and standalone plugin (JUCE 9) built on the C++ recommender in
`../cpp`; `INSTALL.md` explains how to install the released packages. It
listens to a track, decides every parameter of a Butterworth high-pass filter for it (whether to
filter, the corner and the slope, from 6 to 48 dB/oct) with the same method and report as the
command-line tool, and applies those decisions.

A fixed 20 Hz high-pass at 24 dB/oct (fourth-order Butterworth) is always on, ahead of everything:
what lies below 20 Hz is not useful content. The analysis runs on the signal after that filter, so
infrasonic rumble and DC offset neither switch the recommended filter on nor pull the protected note
down, and the recommended filter, when it is on, comes after the fixed one. The two share the loss
budget at the protected frequency. The fixed filter takes 0.13 dB at B0 (30.9 Hz), 0.01 dB at E1 and
24 dB at 10 Hz.

Version 0.3. It adds the fixed 20 Hz filter and is published by Atonal (manufacturer code `Atnl`,
plugin code `Hpfr`); 0.2 used other codes, so hosts list 0.3 as a new plugin and sessions saved with 0.2
do not reopen it. The interface and the notes of the recommender's report are in English. The slope
parameter (`slope`, 6 to 48 dB/oct) replaced the 0.1 parameter `order` (12 to 24 dB/oct); sessions saved
with 0.1 open with the default slope.

## Use

The plugin is installed by the build in `~/Library/Audio/Plug-Ins/VST3` and
`~/Library/Audio/Plug-Ins/Components` (company Atonal, manufacturer code `Atnl`,
plugin code `Hpfr`). In a DAW the fastest workflow is:

1. Insert the plugin on the tracks to clean. A freshly inserted plugin is already listening.
2. Play the song once from start to end, or run an offline render (in REAPER, a dry-run render):
   the plugin records its input, before its own filter, while the transport plays or while the host
   renders, and the render can go faster than real time.
3. Stop. One second after the audio stops arriving, each plugin analyses what it heard (one analysis
   at a time across instances). With **Apply when done** on, the default, the
   corner, the slope and the on/off decision go straight into the plugin's parameters; when there is
   nothing measurable to remove, the recommended filter stays off with the safe corner ready, and the
   track still goes through the fixed 20 Hz filter.

The window shows one line per parameter with the reason for each decision: whether to filter (only
when there is low-frequency noise above the usefulness threshold or a DC offset), the corner (the
highest that keeps the protected frequency within the allowed loss, counting the fixed filter's), the slope (the gentlest that
brings that noise down to the threshold, up to 48 dB/oct on a note and 24 dB/oct on a spectral edge)
and the curve (always Butterworth). It also shows the loss, phase and group delay at the protected
frequency, and any notes (mains hum, rumble it cannot remove). **Listen again**
starts a new capture; **Analyse now** analyses what has been heard so far; **Apply**
applies the last analysis by hand. Corner, slope and on/off stay editable and automatable.

The last analysis is saved with the session, and a session that brings its analysis does not listen
again. In the standalone application there is no transport: it listens only after **Listen** is
pressed. Recording stops by itself after 10 minutes. A track whose two channels are identical (a mono
track on a stereo bus) is stored and analysed as mono.

A host that renders at a sample rate other than the session's prepares the plugin again when the
render starts and when it ends. When that happens while the plugin is listening, what it has heard
is analysed straight away at the rate it was heard at, if it is at least 3 seconds long; if it is
shorter, it is discarded and the plugin keeps listening at the new rate.

## How it decides

### 1. Finding the lowest note

1. **Pitch every 10 ms.** The YIN algorithm measures how long the waveform takes to repeat; that
   period is the pitch at that moment. It looks between 27.5 Hz (A0) and 1 kHz and ignores
   near-silent moments.
2. **Each reading is checked against the spectrum** of the same moment, because YIN can be wrong
   by an octave. A note is accepted only if it has its own peak standing out from the noise around
   it, is not too far below the strongest harmonic, and shows a peak at twice or three times its
   frequency (it has harmonics).
3. **Things that look like notes are discarded.** Mains hum (50 or 60 Hz and its multiples), and
   rumble: a constant, nearly periodic noise that YIN mistakes for a note. Rumble is recognised
   because it is there all the time; a real note in the same region survives if it rises above that
   noise while it sounds.
4. **Enough evidence.** The lowest note is the lowest pitch found in at least 12 readings (about
   120 ms), with a nearly continuous run of 6. This filters out isolated errors; the cost is that
   low notes shorter than about 200 ms may not count.
5. **What kind of signal it is.** If the moments with a note carry at least a quarter of the energy
   (bass, guitar, voice, saxophone), the lowest note is protected. If not (drums, room), the lower
   edge of the spectrum is protected: the point where the signal has fallen 30 dB below its
   strongest part.

### 2. Choosing the right octave of the note

- **False low reading.** E2 and B2 played together can make YIN read 41 Hz (E1), the period both
  notes share, although E1 is not played. 41 Hz has no peak of its own, so the reading moves up to
  the first multiple that is a credible note: E2 (82.4 Hz).
- **False high reading.** On a miked bass the second harmonic can be stronger than the
  fundamental, and YIN reads E2 while E1 sounds. Going down an octave is accepted only if E1 has its
  own credible peak and YIN itself confirms that the wave repeats better at the double period.
- In short: the lowest octave with real evidence in the spectrum, and never lower without proof.

### 3. How far below the note the cut goes

There is no fixed distance. The corner is the highest one that, together with the fixed 20 Hz filter,
takes at most 1 dB from the lowest note, and that depends on the slope: a gentle filter has to sit further below the note, a steep one
can come closer. For a bass in E1 (41.2 Hz):

| Slope | Corner | Distance below E1 | Loss at E1 (both filters) |
| --- | --- | --- | --- |
| 6 dB/oct | 20 Hz | about 1 octave | 0.93 dB |
| 12 dB/oct | 28 Hz | about half an octave | 0.85 dB |
| 24 dB/oct | 31.5 Hz | about 0.4 octave | 0.49 dB |
| 48 dB/oct | 35.5 Hz | about 0.2 octave | 0.40 dB |

The corner is always rounded down to a standard value (the R20 series: 20, 22.4, 25, 28, 31.5,
35.5 Hz...), so rounding never takes more from the note.

The slope is the gentlest one that brings the low-frequency noise below the note down to 30 dB under
the track's strongest part. The half octave just under the note is left out of that check, because
the note's own spill lands there. When there is no noise to bring down, the recommendation is to
leave the filter off.

For unpitched sources (drums, cymbals, room) the corner goes right on the spectral edge, with 3 dB
of loss there. If constant noise reaches that edge, the measured edge is not reliable and the budget
goes back to 1 dB, with more margin. The slope stays at 24 dB/oct or less, because with a steeper
filter any error in measuring the edge becomes more loss.

## Real-time design

- The audio thread only copies samples into a lock-free queue whose memory is reserved in
  `prepareToPlay`, and runs the filter. Accumulating the recording and running `hpf::recommend`
  happen on a separate thread.
- `Source/HighPass.h` is the same Butterworth filter as the recommender (bilinear transform with
  prewarping, as `scipy.signal.butter`), but its design does not allocate memory. The corner is
  therefore smoothed over 50 ms by recomputing the coefficients every 32 samples, and switching the
  filter on or off crossfades with the unfiltered signal over 20 ms.
- `tests/test_highpass.cpp` checks the filter against the recommender: frequency response against
  `hpf::design::sos`, and output processed in random block sizes against `hpf::apply_hpf`.
- `tests/rtsan_highpass.cpp` checks with RealtimeSanitizer that designing the filter and processing a
  block inside a real-time function neither allocate nor block.
- `tests/test_analyser.cpp` feeds a track to the listening path block by block, as the audio thread
  does, and checks that the report is identical to `hpf::recommend` on the same signal, that the
  analysis starts by itself when the audio stops arriving, that identical channels are analysed
  as mono, and that a change of sample rate by the host neither stops the listening nor throws away
  what was heard.
- `tests/test_host.cpp` loads the built VST3 in a minimal host with a simulated transport and checks
  the DAW workflow: listening from insertion, analysis when the transport stops and after an offline
  render (also a 48 kHz render from a host running at 44.1 kHz), the applied corner, slope and
  on/off equal to `hpf::recommend` on the same signal, no listening without a transport unless asked,
  a saved session coming back without listening again, and an output equal to the core's chain (the
  fixed 20 Hz filter, with the recommended filter off and on) within -80 dB.
- In a host that delivers audio faster than real time without declaring an offline render, the
  capture queue (about one second) fills and samples are lost; the report says how many seconds.

## Build

Needs CMake 3.22, a C++17 compiler and JUCE (by default in `/Applications/JUCE-9.0.6`; it also builds
with JUCE 8.0.12).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build
```

`-DJUCE_DIR=...` and `-DHPF_CORE_DIR=...` point to other copies of JUCE and of
`cpp` (the core). The plugins are written to `build/HPFRecommender_artefacts/Release/`.

RealtimeSanitizer needs Clang from LLVM (Apple Clang does not accept `-fsanitize=realtime`):

```bash
/opt/homebrew/opt/llvm/bin/clang++ -std=c++17 -fsanitize=realtime -ISource tests/rtsan_highpass.cpp -o build/rtsan_highpass
./build/rtsan_highpass
```

## Limitations of version 0.3

- The track has to be played once, in real time or through an offline render, to be analysed. ARA
  (Audio Random Access), which lets a plugin read a whole clip without playback in hosts that support
  it, is the way to remove that step.
- Validated with pluginval (VST3 and AU, strictness level 5, with its GUI tests) and with `auval`
  (AU), with JUCE 9.0.6. The DAW workflow is tested in a simulated host (`tests/test_host.cpp`)
  and in REAPER 7.86 (macOS, arm64) through offline renders only (`tests/reaper/`), with the version
  before the fixed 20 Hz filter, which has to be tested there again: with the AU and the
  VST3, the render with the plugin just inserted equals the input sample for sample; after the render
  the plugin applied 50.0 Hz, 6 dB/oct, filter on, the same as `hpf_recommend`, and the next render
  matches the input filtered by the core's design within -116.3 dB. Playback through an audio
  interface, other DAWs and a change of sample rate inside a DAW (REAPER ran the plugin at 48 kHz and
  resampled a 44.1 kHz render) have not been tested.
- Each listening instance keeps the audio it heard until it analyses it: about 92 MB per instance for
  four minutes of stereo audio at 48 kHz (half for a mono track).
- AAX (Pro Tools) is not built yet.
- The plugin is licensed under the GNU Affero General Public License v3 (`LICENSE.md`), one of the two
  licences under which JUCE 9 may be used; the recommender core it builds on (`hpf-recommender`) stays
  under the MIT licence. The released builds are not code-signed (no Apple Developer ID or Windows certificate yet);
  the Windows and Linux builds are compiled and unit-tested on GitHub Actions but have not been tried
  in a DAW.
