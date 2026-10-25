# mkv-audio-sync

With good audio, use `crosscorrelation.py`, it is more accurate in these cases.

With bad audio, or audio with lots of noise, use `fft.py`.

To cross-check the result in Audacity, use `extract_audio_from_mkv.sh` to get the WAV files of each track.

Install FFmpeg and Python 3.10 or newer, then run `python3 -m pip install -r requirements.txt`.

```bash
python3 crosscorrelation.py reference.mkv other.mkv
python3 crosscorrelation.py reference.mkv other.mkv --lang eng
python3 fft.py reference.mkv other.mkv
```

The delay applies to the second file's audio. These scripts measure one constant offset, not clock drift or different edits. Check the result by ear before muxing. The default 30-minute segment uses substantial memory; `--seg_length` limits it in `crosscorrelation.py`.

`examples/make-example.sh` generates a synthetic pair with a 500 ms delay. Run `python3 -m pytest` for the tests.

MIT, see [LICENSE](LICENSE).
