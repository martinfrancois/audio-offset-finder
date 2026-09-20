#!/usr/bin/env bash
# Generate two one-minute MKV files with the same pink-noise audio track.
# In delayed.mkv the audio starts 500 ms later than in reference.mkv.
#
# Usage: examples/make-example.sh [output-folder]
#
# The video is a static gray 64x64 frame. Only encoders built into every
# ffmpeg build are used (ffv1, flac, pcm_s16le).
set -euo pipefail

out="${1:-examples}"
mkdir -p "$out"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "anoisesrc=color=pink:seed=42:duration=60:sample_rate=48000" \
  -c:a pcm_s16le "$out/noise.wav"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "color=c=gray:size=64x64:rate=5:duration=60" \
  -i "$out/noise.wav" \
  -map 0:v -map 1:a -c:v ffv1 -c:a flac \
  -metadata:s:a:0 language=eng -shortest "$out/reference.mkv"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "color=c=gray:size=64x64:rate=5:duration=60" \
  -i "$out/noise.wav" \
  -af "adelay=500|500" \
  -map 0:v -map 1:a -c:v ffv1 -c:a flac \
  -metadata:s:a:0 language=eng -shortest "$out/delayed.mkv"

rm -f "$out/noise.wav"
echo "Wrote $out/reference.mkv and $out/delayed.mkv (audio in delayed.mkv starts 500 ms later)"
