#!/usr/bin/env bash
# Generate two one-minute MKV files with the same pink-noise audio track.
# In delayed.mkv the audio starts delay_ms (set below) later than in reference.mkv.
#
# Usage: examples/make-example.sh [output-folder]
#
# The video is a static gray 64x64 frame. Only encoders built into every
# ffmpeg build are used (ffv1, flac, pcm_s16le).
set -euo pipefail

# README.md and the CI workflow check the delay the scripts print for this value.
delay_ms=500

out="${1:-examples}"
mkdir -p "$out"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "anoisesrc=color=pink:seed=42:duration=60:sample_rate=48000" \
  -c:a pcm_s16le "$out/noise.wav"

# Mux the noise under a gray video into $1; further arguments go to ffmpeg as output options.
mux() {
  local target="$1"
  shift
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "color=c=gray:size=64x64:rate=5:duration=60" \
    -i "$out/noise.wav" \
    "$@" \
    -map 0:v -map 1:a -c:v ffv1 -c:a flac \
    -metadata:s:a:0 language=eng -shortest "$target"
}

mux "$out/reference.mkv"
mux "$out/delayed.mkv" -af "adelay=${delay_ms}|${delay_ms}"

rm -f "$out/noise.wav"
echo "Wrote $out/reference.mkv and $out/delayed.mkv (audio in delayed.mkv starts ${delay_ms} ms later)"
