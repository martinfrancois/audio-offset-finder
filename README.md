# mkv-audio-sync

With good audio, use `crosscorrelation.py`, it is more accurate in these cases.

With bad audio, or audio with lots of noise, use `fft.py`.

If you want to cross-check, if it is correct, you can use `extract_audio_from_mkv` to get the wav files of each track to then compare in Audacity.