# HPF! by Atonal

HPF! listens to a track and sets its high-pass filter: whether to filter, the cutoff and the slope.
Insert it on a track, play the track once from start to end (or run an offline render) and stop; it
analyses what it heard and applies the result. Every package also has a standalone application.

## macOS (11 Big Sur or later, Apple Silicon and Intel)

1. Copy `HPF!.vst3` to `~/Library/Audio/Plug-Ins/VST3/`.
2. Copy `HPF!.component` to `~/Library/Audio/Plug-Ins/Components/` (Audio Unit, for Logic Pro and
   GarageBand).
3. Restart the DAW or rescan plug-ins.

The builds are not notarized by Apple. If macOS blocks them, run in Terminal:

    xattr -dr com.apple.quarantine ~/Library/Audio/Plug-Ins/VST3/HPF!.vst3
    xattr -dr com.apple.quarantine ~/Library/Audio/Plug-Ins/Components/HPF!.component

## Windows (10 or 11, 64-bit)

Copy the `HPF!.vst3` folder to `C:\Program Files\Common Files\VST3\` and rescan plug-ins in the DAW.
Windows may warn that the publisher is unknown, because the files are not code-signed.

## Linux (64-bit, glibc 2.35 or later, for example Ubuntu 22.04)

Copy the `HPF!.vst3` folder to `~/.vst3/` and rescan plug-ins in the DAW (REAPER, Bitwig, Ardour and
other VST3 hosts).

## Source and licence

Source code, tests and the paper behind the method: https://github.com/marianoalvarez130996/hpf-recommender

HPF! is free software under the GNU Affero General Public License v3 (`LICENSE.md`).
