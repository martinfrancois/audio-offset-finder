#!/bin/bash
# extract_audio_from_mkv.sh
#
# Usage: ./extract_audio_from_mkv.sh <input_file> [output_prefix]
#   <input_file>    : The MKV (or other) file from which to extract audio.
#   [output_prefix] : Optional. The prefix for the output files. Default is "track_".
#
# This script uses ffprobe to count the audio streams and then loops over them,
# extracting each one as a mono PCM 16-bit WAV file.

# Check that an input file was provided.
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <input_file> [output_prefix]"
  exit 1
fi

input_file="$1"
output_prefix="${2:-track_}"

# Count the number of audio streams in the file.
num_tracks=$(ffprobe -v error -select_streams a \
  -show_entries stream=index \
  -of csv=p=0 "$input_file" | wc -l)

echo "Found $num_tracks audio stream(s) in '$input_file'."

# Loop over each audio stream and extract it.
for (( i=0; i<num_tracks; i++ )); do
  # Format the output file name, e.g. track_001.wav, track_002.wav, etc.
  out=$(printf "%s%03d.wav" "$output_prefix" $((i+1)))
  echo "Extracting audio stream $i to $out..."
  ffmpeg -i "$input_file" -map 0:a:$i -ac 1 -c:a pcm_s16le "$out"
done
