# audio-offset-finder

Finds the offset between the audio tracks of two MKV files, so the track can be muxed in with the right `mkvmerge --sync` delay.

## What it does

Two files of the same film often carry audio tracks that are shifted against
each other by some tens or hundreds of milliseconds. Both scripts extract a
segment from the middle of each file with ffmpeg, compare the two segments in
Python and print the delay in milliseconds to pass to `mkvmerge --sync` for
the second file's audio track. A negative delay means the second file's audio
starts later and must be advanced.

- `crosscorrelation.py` cross-correlates the raw samples. Use it with clean
  audio; it resolves the offset to one sample.
- `fft.py` matches the strongest spectral peaks per time window. Use it with
  noisy audio. It resolves the offset to one FFT window, 1024 samples, which
  is 21 ms at 48 kHz.
- `extract_audio_from_mkv.sh` writes every audio track of a file to a mono
  16-bit WAV file, for checking a result by hand in an audio editor.

## Requirements

- Python 3.10 or newer with the pinned `numpy` and `scipy` from
  `requirements.txt`.
- `ffmpeg` and `ffprobe` on the `PATH`.

Install ffmpeg:

```bash
# Debian, Ubuntu
sudo apt-get update && sudo apt-get install -y ffmpeg

# Fedora (ffmpeg is in the RPM Fusion free repository)
sudo dnf install -y https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install -y ffmpeg --allowerasing

# macOS (Homebrew)
brew install ffmpeg
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

```bash
python3 crosscorrelation.py reference.mkv other.mkv
python3 crosscorrelation.py reference.mkv other.mkv --seg_length 600 --sr 16000
python3 fft.py reference.mkv other.mkv
./extract_audio_from_mkv.sh other.mkv other_
```

The first file is the one whose video you keep. The printed delay applies to
the audio track of the second file.

`crosscorrelation.py` takes a 1800 s segment at 48 kHz by default; shorter
files are used whole. `--seg_length` and `--sr` change that. `fft.py` uses the
same 1800 s and 48 kHz and has no options. It writes `<name>_middle.wav` for
both inputs into the current directory and leaves them there.

`extract_audio_from_mkv.sh` writes `other_001.wav`, `other_002.wav` and so on,
one per audio stream.

## Example

`examples/make-example.sh` builds two one-minute MKV files with ffmpeg. Both
carry the same seeded pink-noise track over a static gray frame; in
`delayed.mkv` the audio starts 500 ms later. Nothing is recorded, so the files
are not committed and are ignored by git.

```bash
examples/make-example.sh
python3 crosscorrelation.py examples/reference.mkv examples/delayed.mkv
```

The last lines of the output:

```
=== Cross-correlation results ===
Best lag (in samples): 24000
Maximum correlation value: 116569461018625.23
Computed sync delay to apply to file2's audio: -500 ms

In mkvtoolnix (mkvmerge), you would use:
    --sync <trackID>:-500
(A negative delay means file2's audio should be advanced relative to file1.)
```

24000 samples at 48 kHz is exactly the 500 ms that the generator inserted.
Two `WavFileWarning: Reached EOF prematurely` lines from scipy precede this
output. ffmpeg streams the WAV to a pipe and cannot write the real length
into the header; the data is complete and the warning is harmless.

The same pair through `fft.py`:

```bash
python3 fft.py examples/reference.mkv examples/delayed.mkv
```

```
=== Alignment Results ===
Computed delay: -0.4907 seconds (-491 ms)
```

The 9 ms error is the window resolution: 500 ms is 23.4 windows of 21.3 ms,
and the script reports whole windows.

Run the tests with `python -m pytest`. They synthesize signals with numpy and
need no ffmpeg.

## Limitations

- Cross-correlation finds one constant offset. It cannot detect drift, where
  the offset grows over the file because the two sources run at slightly
  different speeds, and it gives no useful result when the audio was cut
  differently, for example one version with a scene removed. The segment is
  always taken from the middle of the first file's track; `--seg_length`
  only sets how long it is.
- Cross-correlation of raw samples assumes the two tracks contain the same
  sound. Different mixes, different languages or heavy noise reduction lower
  the correlation peak until a wrong lag wins. Only the peak position is
  printed, so check the "Maximum correlation value" against a run on a pair
  that is known to be in sync, or verify the result by ear with the extracted
  WAV files.
- Periodic audio, such as a steady music loop, produces several equally strong
  peaks one period apart. The script reports whichever is largest.
- The whole segment is held in memory. At the default 1800 s and 48 kHz each
  segment is 86 million samples, about 700 MB as float64, before scipy's FFT
  buffers come on top.
- `fft.py` reports whole FFT windows, so its result is off by up to 21 ms.
- `--lang` in `crosscorrelation.py` does not work: the ffprobe query in
  `get_audio_tracks` never returns the language tag, every track is reported
  as `und`, and any `--lang` value other than `und` exits with "No audio track
  found". Without `--lang` the first audio track of each file is used.
- Both scripts exit on the first ffmpeg or ffprobe error and print its
  message.

## License

MIT, see `LICENSE`.
