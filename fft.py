#!/usr/bin/env python3
"""
This script extracts an audio segment (30 minutes by default) from the middle of two MKV files,
processes each segment by computing FFT bins and grouping peaks into boxes,
and then finds matching frequency “features” between the two. It computes the most
common time offset (in seconds and milliseconds) that best aligns file2’s audio with
file1’s audio. You can then apply this delay (using mkvtoolnix’s --sync option) when muxing.

Usage:
    python3 fft.py file1.mkv file2.mkv
"""

import argparse
import subprocess
import math
import numpy as np
import scipy.io.wavfile
import os
import sys

DEFAULT_SEGMENT_LENGTH = 1800  # seconds
SAMPLE_RATE = 48000  # Hz
DEFAULT_FFT_BIN_SIZE = 1024
DEFAULT_OVERLAP = 0
DEFAULT_BOX_HEIGHT = 512
DEFAULT_BOX_WIDTH = 43
DEFAULT_SAMPLES_PER_BOX = 7

def get_duration(video_file):
    """
    Uses ffprobe to get the duration (in seconds) of the given video file.
    """
    import subprocess
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_file
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        # Split the output into lines and take the last non-empty line.
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        duration_str = lines[-1]
        return float(duration_str)
    except subprocess.CalledProcessError as e:
        print(f"Error getting duration for {video_file}: {e.output}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error converting duration output to float: '{output}'. {e}")
        sys.exit(1)

def extract_audio(video_file, start_time, seg_length, output_file):
    """
    Extracts a segment of the audio from video_file using ffmpeg.
    The segment begins at start_time (in seconds) and lasts for seg_length seconds.
    The audio is converted to mono WAV at SAMPLE_RATE.
    """
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-t', str(seg_length),
        '-i', video_file,
        '-vn',
        '-ac', '1',
        '-ar', str(SAMPLE_RATE),
        '-f', 'wav',
        output_file
    ]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_file}:\n{e.output}")
        sys.exit(1)
    return output_file

def read_audio(audio_file):
    """
    Reads a WAV audio file using scipy.io.wavfile.
    Returns a tuple (data, sample_rate).
    """
    try:
        rate, data = scipy.io.wavfile.read(audio_file)
        return data, rate
    except Exception as e:
        print(f"Error reading audio file {audio_file}: {e}")
        sys.exit(1)

def fourier(sample):
    """
    Computes the one-dimensional discrete Fourier Transform on a chunk of audio data.
    Returns a list of magnitudes (rounded to 2 decimals) for the first half of the FFT bins.
    """
    fft_data = np.fft.fft(sample)
    mag = []
    half = len(fft_data) // 2
    for i in range(half):
        r = fft_data[i].real ** 2
        j = fft_data[i].imag ** 2
        mag.append(round(math.sqrt(r + j), 2))
    return mag

def make_horiz_bins(data, fft_bin_size, overlap, box_height):
    """
    Processes the audio data in successive windows of length fft_bin_size.
    For each FFT window, compute the FFT magnitude and group each frequency
    bin into a horizontal “box” determined by box_height.
    Returns a dictionary mapping box_y indices to lists of tuples:
        (intensity, time_coord, frequency_bin_index)
    """
    horiz_bins = {}
    # Process the first window
    sample_data = data[0:fft_bin_size]
    if len(sample_data) == fft_bin_size:
        intensities = fourier(sample_data)
        for i in range(len(intensities)):
            box_y = i // box_height
            if box_y in horiz_bins:
                horiz_bins[box_y].append((intensities[i], 0, i))
            else:
                horiz_bins[box_y] = [(intensities[i], 0, i)]
    # Process subsequent windows
    x_coord_counter = 1
    step = fft_bin_size - overlap
    for j in range(step, len(data) - fft_bin_size + 1, step):
        sample_data = data[j:j + fft_bin_size]
        if len(sample_data) == fft_bin_size:
            intensities = fourier(sample_data)
            for k in range(len(intensities)):
                box_y = k // box_height
                if box_y in horiz_bins:
                    horiz_bins[box_y].append((intensities[k], x_coord_counter, k))
                else:
                    horiz_bins[box_y] = [(intensities[k], x_coord_counter, k)]
        x_coord_counter += 1
    return horiz_bins

def make_vert_bins(horiz_bins, box_width):
    """
    Groups the horizontal bins into vertical “boxes” by dividing the time coordinate by box_width.
    Returns a dictionary with keys (box_x, box_y) and values as lists of tuples.
    """
    boxes = {}
    for box_y in horiz_bins:
        for tup in horiz_bins[box_y]:
            box_x = tup[1] // box_width
            key = (box_x, box_y)
            if key in boxes:
                boxes[key].append(tup)
            else:
                boxes[key] = [tup]
    return boxes

def find_bin_max(boxes, maxes_per_box):
    """
    For each vertical box (grouped by time and frequency), select up to maxes_per_box points
    with the highest intensity.
    Returns a dictionary mapping frequency bin indices to a list of time coordinates.
    """
    freqs_dict = {}
    for key in boxes:
        max_intensities = []
        for item in boxes[key]:
            if len(max_intensities) < maxes_per_box:
                max_intensities.append(item)
            else:
                # Find the point with the smallest intensity so far
                min_item = min(max_intensities, key=lambda x: x[0])
                if item[0] > min_item[0]:
                    max_intensities.remove(min_item)
                    max_intensities.append(item)
        for tup in max_intensities:
            freq_index = tup[2]   # frequency bin index
            time_coord = tup[1]   # time coordinate (x)
            if freq_index in freqs_dict:
                freqs_dict[freq_index].append(time_coord)
            else:
                freqs_dict[freq_index] = [time_coord]
    return freqs_dict

def find_freq_pairs(freqs_dict_orig, freqs_dict_sample):
    """
    Finds pairs of matching frequency bin indices between the reference (orig) and sample dictionaries.
    For every frequency bin that occurs in both, pairs each time coordinate from sample with each
    time coordinate from orig.
    Returns a list of (sample_time, orig_time) pairs.
    """
    time_pairs = []
    for freq in freqs_dict_sample:
        if freq in freqs_dict_orig:
            for t_sample in freqs_dict_sample[freq]:
                for t_orig in freqs_dict_orig[freq]:
                    time_pairs.append((t_sample, t_orig))
    return time_pairs

def find_delay(time_pairs):
    """
    From the list of time coordinate pairs, computes the difference (delta) for each pair.
    Tallies the frequency of each delta and returns the one that occurs most often.
    (This delta is in units of FFT window “time coordinates”.)
    """
    t_diffs = {}
    for pair in time_pairs:
        delta_t = pair[0] - pair[1]
        if delta_t in t_diffs:
            t_diffs[delta_t] += 1
        else:
            t_diffs[delta_t] = 1
    if not t_diffs:
        print("No frequency pairs found to compute delay.")
        sys.exit(1)
    t_diffs_sorted = sorted(t_diffs.items(), key=lambda x: x[1])
    time_delay = t_diffs_sorted[-1][0]
    return time_delay

def find_peaks(data, fft_bin_size, overlap, box_height, box_width, samples_per_box):
    """
    Computes the FFT windows of the audio data, groups them into boxes and keeps the
    strongest points per box.
    Returns a dictionary mapping frequency bin indices to lists of time coordinates.
    """
    horiz_bins = make_horiz_bins(data, fft_bin_size, overlap, box_height)
    boxes = make_vert_bins(horiz_bins, box_width)
    return find_bin_max(boxes, samples_per_box)

def compute_delay(audio1, audio2, rate, fft_bin_size=DEFAULT_FFT_BIN_SIZE,
                  overlap=DEFAULT_OVERLAP, box_height=DEFAULT_BOX_HEIGHT,
                  box_width=DEFAULT_BOX_WIDTH, samples_per_box=DEFAULT_SAMPLES_PER_BOX):
    """
    Finds the delay (in seconds) to apply to audio2 so that it aligns with audio1.
    Both inputs are 1-D arrays of samples at the given rate. A negative delay means
    audio2 starts later and must be advanced.
    """
    peaks1 = find_peaks(audio1, fft_bin_size, overlap, box_height, box_width, samples_per_box)
    peaks2 = find_peaks(audio2, fft_bin_size, overlap, box_height, box_width, samples_per_box)
    delay_in_bins = find_delay(find_freq_pairs(peaks1, peaks2))

    # Each x coordinate corresponds to one FFT window step.
    time_per_x = (fft_bin_size - overlap) / rate
    # mkvmerge expects the correction, which is the opposite of the measured lag.
    return -delay_in_bins * time_per_x

def align(video1, video2, segment_length=DEFAULT_SEGMENT_LENGTH,
          fft_bin_size=DEFAULT_FFT_BIN_SIZE, overlap=DEFAULT_OVERLAP,
          box_height=DEFAULT_BOX_HEIGHT, box_width=DEFAULT_BOX_WIDTH,
          samples_per_box=DEFAULT_SAMPLES_PER_BOX):
    """
    Main alignment function:
      1. Determines the duration of the first file and computes a start time so that a
         segment of segment_length seconds is extracted from the middle of both files.
      2. Extracts the audio segments (as mono WAV at SAMPLE_RATE) from each file.
      3. Passes both segments to compute_delay, which finds the strongest frequency
         peaks, matches them between file1 (reference) and file2, and converts the
         most common time offset (in units of FFT windows) into seconds.
    """
    # Get durations and compute start times for the middle segment.
    dur1 = get_duration(video1)
    start = max((dur1 / 2) - (segment_length / 2), 0)

    # Generate temporary WAV file names based on the input file names.
    base1 = os.path.splitext(os.path.basename(video1))[0]
    base2 = os.path.splitext(os.path.basename(video2))[0]
    wav1 = base1 + "_middle.wav"
    wav2 = base2 + "_middle.wav"

    print(f"Extracting {segment_length:.0f} s segment from {video1} starting at {start:.2f}s -> {wav1}")
    extract_audio(video1, start, segment_length, wav1)
    print(f"Extracting {segment_length:.0f} s segment from {video2} starting at {start:.2f}s -> {wav2}")
    extract_audio(video2, start, segment_length, wav2)

    # Read the extracted audio data.
    raw_audio1, rate1 = read_audio(wav1)
    raw_audio2, rate2 = read_audio(wav2)
    if rate1 != rate2:
        print("Sample rates do not match between files.")
        sys.exit(1)
    rate = rate1
    print(f"Audio sample rate: {rate} Hz")

    delay_seconds = compute_delay(raw_audio1, raw_audio2, rate, fft_bin_size, overlap,
                                  box_height, box_width, samples_per_box)
    delay_ms = delay_seconds * 1000

    print("\n=== Alignment Results ===")
    print(f"Computed delay: {delay_seconds:.4f} seconds ({delay_ms:.0f} ms)")
    print("Apply this delay to file2's audio track (e.g. with mkvmerge’s --sync option).")

    return delay_seconds

def main():
    parser = argparse.ArgumentParser(
        description="Align the audio of two MKV files by matching a segment from the middle using FFT bin analysis."
    )
    parser.add_argument("file1", help="First MKV file (reference)")
    parser.add_argument("file2", help="Second MKV file (to be aligned)")
    args = parser.parse_args()

    align(args.file1, args.file2)

if __name__ == "__main__":
    main()
