#!/usr/bin/env python3
"""
This script calculates the sync delay you should apply to file2's audio track
when muxing it into file1 (which has the video you wish to keep). Although each
file is internally in sync (audio and video), the audio streams from the two
sources may be offset relative to each other. This script extracts an audio
segment (30 minutes by default, or the whole track when it is shorter) from the
middle of the chosen audio track in each file and then uses cross-correlation
to compute the offset between them.

Usage:
    python3 crosscorrelation.py file1.mkv file2.mkv [--lang eng,fra]

If --lang is provided, only audio streams whose language tag matches one of the
comma-separated codes are considered. Otherwise, the first audio track from each file is used.

The computed delay (in milliseconds) is the value you will supply to mkvtoolnix’s
--sync option for file2’s audio track. (A negative delay means that file2's audio
should be advanced relative to file1.)
"""

import argparse
import subprocess
import json
import sys
import re

import numpy as np
from scipy.signal import correlate, correlation_lags

DEFAULT_SEG_LENGTH = 1800  # seconds
DEFAULT_SAMPLE_RATE = 48000  # Hz


def run_ffprobe(cmd):
    """Run an ffprobe command and return its stdout as text. Exit on error."""
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running ffprobe: {' '.join(cmd)}\n{e.stderr}")
        sys.exit(1)


LANGUAGE_ALIASES = {
    "eng": {"eng", "en"},
    "en": {"eng", "en"},
    "ger": {"ger", "deu", "de"},
    "deu": {"ger", "deu", "de"},
    "de": {"ger", "deu", "de"},
    "fra": {"fra", "fre", "fr"},
    "fre": {"fra", "fre", "fr"},
    "fr": {"fra", "fre", "fr"},
    "spa": {"spa", "es"},
    "es": {"spa", "es"},
    "ita": {"ita", "it"},
    "it": {"ita", "it"},
    "jpn": {"jpn", "ja"},
    "ja": {"jpn", "ja"},
    "zho": {"zho", "chi", "zh"},
    "chi": {"zho", "chi", "zh"},
    "zh": {"zho", "chi", "zh"},
    "kor": {"kor", "ko"},
    "ko": {"kor", "ko"},
    "por": {"por", "pt"},
    "pt": {"por", "pt"},
    "rus": {"rus", "ru"},
    "ru": {"rus", "ru"},
    "nld": {"nld", "dut", "nl"},
    "dut": {"nld", "dut", "nl"},
    "nl": {"nld", "dut", "nl"},
}

LANGUAGE_PREFIXES = {
    "eng": ["english"],
    "en": ["english"],
    "ger": ["german", "deutsch"],
    "deu": ["german", "deutsch"],
    "de": ["german", "deutsch"],
    "fra": ["french", "francais"],
    "fre": ["french", "francais"],
    "fr": ["french", "francais"],
    "spa": ["spanish", "espanol"],
    "es": ["spanish", "espanol"],
    "ita": ["italian", "italiano"],
    "it": ["italian", "italiano"],
    "jpn": ["japanese"],
    "ja": ["japanese"],
    "zho": ["chinese"],
    "chi": ["chinese"],
    "zh": ["chinese"],
    "kor": ["korean"],
    "ko": ["korean"],
    "por": ["portuguese"],
    "pt": ["portuguese"],
    "rus": ["russian"],
    "ru": ["russian"],
    "nld": ["dutch"],
    "dut": ["dutch"],
    "nl": ["dutch"],
}


def normalize_language_code(lang):
    if not lang:
        return "und"
    return str(lang).strip().lower()


def language_aliases(lang):
    norm = normalize_language_code(lang)
    return LANGUAGE_ALIASES.get(norm, {norm})


def language_matches(track_lang, desired_langs):
    if not desired_langs:
        return True
    track_aliases = language_aliases(track_lang)
    for desired in desired_langs:
        if track_aliases & language_aliases(desired):
            return True
    return False


def normalize_commercial_key(name):
    if not name:
        return ""
    return re.sub(r"\s+", " ", name).strip().casefold()


def strip_language_prefix(title, language):
    if not title:
        return title
    cleaned = title.strip()
    lang = normalize_language_code(language)
    prefixes = set(language_aliases(lang))
    for alias in list(prefixes):
        prefixes.update(LANGUAGE_PREFIXES.get(alias, []))
    for prefix in sorted(prefixes, key=len, reverse=True):
        if not prefix:
            continue
        pattern = r"^" + re.escape(prefix) + r"\b[\s:-]*"
        if re.match(pattern, cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
            break
    return cleaned


def derive_commercial_name(title, language, codec_long_name, codec_name):
    name = title.strip() if title else ""
    if name:
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
        name = strip_language_prefix(name, language)
    if not name:
        name = codec_long_name or codec_name or "unknown"
    name = re.sub(r"\s+", " ", name).strip()
    return name

def get_file_duration(filename):
    """
    Returns the overall duration (in seconds) of the file using the container's metadata.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        filename
    ]
    output = run_ffprobe(cmd)
    try:
        data = json.loads(output)
        duration_str = data.get("format", {}).get("duration")
        if duration_str is None:
            raise ValueError("Duration not found in format metadata.")
        return float(duration_str)
    except (KeyError, ValueError) as e:
        print(f"Error parsing file duration from {filename}: {e}")
        sys.exit(1)


def get_audio_tracks(filename):
    """
    Retrieves a list of audio streams from the given file.
    Each stream is returned as a dict with keys:
      - index      : the actual stream index in the file
      - codec      : codec name (e.g. aac, ac3, truehd)
      - language   : language code (or "und" if undefined)
      - start_time : stream start time (as a string, may be absent)
      - duration   : stream duration (as a string, may be absent)
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,codec_name,codec_long_name,start_time,duration:stream_tags=language,title",
        "-of", "json",
        filename,
    ]
    output = run_ffprobe(cmd)
    try:
        data = json.loads(output)
        streams = data.get("streams", [])
        audio_streams = []
        for stream in streams:
            tags = stream.get("tags", {}) or {}
            tags = {str(k).lower(): v for k, v in tags.items()}
            language = normalize_language_code(tags.get("language"))
            title = tags.get("title")
            codec_long_name = stream.get("codec_long_name")
            commercial_name = derive_commercial_name(
                title, language, codec_long_name, stream.get("codec_name")
            )
            audio_streams.append({
                "index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "codec_long_name": codec_long_name,
                "language": language,
                "title": title,
                "commercial_name": commercial_name,
                "commercial_key": normalize_commercial_key(commercial_name),
                "start_time": stream.get("start_time"),   # may be None
                "duration": stream.get("duration"),         # may be None
            })
        return audio_streams
    except (KeyError, ValueError) as e:
        print(f"Error parsing audio streams from {filename}: {e}")
        sys.exit(1)


def filter_tracks_by_language(tracks, desired_langs):
    if not desired_langs:
        return tracks
    return [t for t in tracks if language_matches(t["language"], desired_langs)]


def describe_track(track, idx):
    title = track.get("title") or "(no title)"
    commercial = track.get("commercial_name") or "unknown"
    codec = track.get("codec") or "unknown"
    language = track.get("language") or "und"
    return f"  [{idx}] stream {track['index']} | {commercial} | {title} | {codec} | {language}"


def print_track_list(file_label, tracks):
    print(f"{file_label}:")
    for i, track in enumerate(tracks, start=1):
        print(describe_track(track, i))


def prompt_for_index(prompt, max_index):
    while True:
        choice = input(prompt).strip()
        try:
            idx = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= idx <= max_index:
            return idx - 1
        print(f"Please choose a number between 1 and {max_index}.")


def prompt_select_track(file_label, tracks, force_prompt=False):
    if not force_prompt and len(tracks) == 1:
        track = tracks[0]
        print(f"{file_label}: using only matching track {track['index']} ({track['commercial_name']})")
        return track
    print_track_list(file_label, tracks)
    idx = prompt_for_index(f"Select audio track for {file_label} (1-{len(tracks)}): ", len(tracks))
    return tracks[idx]


def prompt_select_commercial_name(common_keys, display_names):
    print("Matching commercial names found:")
    keys = sorted(common_keys, key=lambda k: display_names.get(k, k))
    for i, key in enumerate(keys, start=1):
        print(f"  [{i}] {display_names.get(key, key)}")
    idx = prompt_for_index(f"Select commercial name (1-{len(keys)}): ", len(keys))
    return keys[idx]


def select_audio_tracks(file1_label, file2_label, tracks1, tracks2, desired_langs):
    if not tracks1:
        print(f"No audio tracks found in {file1_label}.")
        sys.exit(1)
    if not tracks2:
        print(f"No audio tracks found in {file2_label}.")
        sys.exit(1)

    filtered1 = filter_tracks_by_language(tracks1, desired_langs)
    filtered2 = filter_tracks_by_language(tracks2, desired_langs)

    if desired_langs and not filtered1:
        available = ", ".join(sorted({t["language"] for t in tracks1}))
        print(f"No audio track found in {file1_label} matching languages {desired_langs}.")
        print(f"Available languages: {available}")
        sys.exit(1)
    if desired_langs and not filtered2:
        available = ", ".join(sorted({t["language"] for t in tracks2}))
        print(f"No audio track found in {file2_label} matching languages {desired_langs}.")
        print(f"Available languages: {available}")
        sys.exit(1)

    names1 = {t["commercial_key"] for t in filtered1}
    names2 = {t["commercial_key"] for t in filtered2}
    common_keys = {k for k in names1 & names2 if k}

    if not common_keys:
        print("No matching commercial names across both files.")
        track1 = prompt_select_track(file1_label, filtered1)
        track2 = prompt_select_track(file2_label, filtered2)
        return track1, track2

    display_names = {}
    for track in filtered1 + filtered2:
        display_names.setdefault(track["commercial_key"], track["commercial_name"])

    if len(common_keys) > 1:
        selected_key = prompt_select_commercial_name(common_keys, display_names)
    else:
        selected_key = next(iter(common_keys))

    candidates1 = [t for t in filtered1 if t["commercial_key"] == selected_key]
    candidates2 = [t for t in filtered2 if t["commercial_key"] == selected_key]

    track1 = prompt_select_track(file1_label, candidates1)
    track2 = prompt_select_track(file2_label, candidates2)
    return track1, track2


def compute_segment_start(track_info, file_duration, seg_length=DEFAULT_SEG_LENGTH):
    """
    Given a track info dictionary and the overall file duration, compute the start time
    for a segment extracted from the middle of the track.

    We choose a segment of length seg_length seconds (default DEFAULT_SEG_LENGTH).
    The segment will start at: (start_time + effective_duration/2) - seg_length/2, but not less than 0.

    If the track's own duration is missing, the overall file duration is used as a fallback.
    """
    # Use track start_time if provided, otherwise assume 0.
    start = float(track_info.get("start_time") or 0)
    if track_info.get("duration") is not None:
        dur = float(track_info.get("duration"))
    else:
        # Fallback: assume the audio track spans from 'start' to the overall file duration.
        dur = file_duration - start

    if dur <= 0:
        print("Unable to determine audio track duration.")
        sys.exit(1)
    mid = start + dur / 2
    seg_start = max(0, mid - seg_length / 2)
    # Ensure the segment does not extend beyond the track's effective duration.
    if seg_start + seg_length > start + dur:
        seg_start = max(0, start + dur - seg_length)
    return seg_start


def extract_audio_segment(filename, track_index, seg_start, seg_length=DEFAULT_SEG_LENGTH,
                          sample_rate=DEFAULT_SAMPLE_RATE):
    """
    Uses ffmpeg to extract a segment of the audio track as mono 16-bit PCM.
    Parameters:
      filename    : input file name
      track_index : actual stream index of the desired audio track in the file
      seg_start   : start time (in seconds) at which to extract the segment
      seg_length  : duration (in seconds) of the segment (default DEFAULT_SEG_LENGTH)
      sample_rate : desired sample rate for extraction (default DEFAULT_SAMPLE_RATE)
    Returns:
      sample_rate, numpy array of audio samples.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-xerror",
        "-ss", str(seg_start),
        "-t", str(seg_length),
        "-i", filename,
        "-map", f"0:{track_index}",
        "-ac", "1",              # mono
        "-ar", str(sample_rate), # force sample rate
        "-c:a", "pcm_s16le",
        "-f", "s16le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        detail = e.stderr.decode(errors="replace").strip()
        print(f'cannot extract audio from "{filename}": {detail}', file=sys.stderr)
        sys.exit(1)
    # Raw PCM has no length header for a non-seekable pipe to leave unfinished.
    if not result.stdout or len(result.stdout) % 2:
        print(f'cannot decode audio from "{filename}": expected complete 16-bit PCM samples', file=sys.stderr)
        sys.exit(1)
    return sample_rate, np.frombuffer(result.stdout, dtype="<i2")


def compute_audio_offset(seg1, seg2, sample_rate):
    """
    Given two 1-D numpy arrays (the audio segments from file1 and file2),
    compute the cross-correlation (using FFT convolution) and return the delay (in ms)
    that should be applied to file2’s audio so that it aligns with file1’s audio.

    The computed lag (in samples) is converted to milliseconds.

    Note on sign: If the best correlation occurs at a positive lag (in samples),
    that means that seg2 is delayed relative to seg1. In order to align, file2's
    audio must be advanced (i.e. given a negative delay). Thus, the computed delay is:

        delay_ms = - (lag * 1000 / sample_rate)
    """
    # Remove DC offset
    seg1 = seg1 - np.mean(seg1)
    seg2 = seg2 - np.mean(seg2)

    # Compute full cross-correlation using FFT (efficient for long signals)
    corr = correlate(seg2, seg1, mode="full", method="fft")
    lags = correlation_lags(len(seg2), len(seg1), mode="full")
    best_lag = lags[np.argmax(corr)]
    delay_sec = - best_lag / sample_rate  # negative sign: if seg2 lags, we need to advance it
    delay_ms = delay_sec * 1000
    return delay_ms, best_lag, corr[np.argmax(corr)]


def main():
    parser = argparse.ArgumentParser(
        description="Align the audio tracks of two MKV files by matching a segment from the middle."
    )
    parser.add_argument("file1", help="Primary MKV file (the video container to keep)")
    parser.add_argument("file2", help="Secondary MKV file (contains the audio track to align)")
    parser.add_argument(
        "--lang",
        help=("Comma-separated list of desired language codes (e.g. eng,fra). "
              "If provided, only audio streams with these language tags are considered."),
        default=None
    )
    parser.add_argument(
        "--seg_length",
        help="Length of the audio segment (in seconds) to use for matching (default: %(default)s)",
        type=float,
        default=DEFAULT_SEG_LENGTH,
    )
    parser.add_argument(
        "--sr",
        help="Sample rate (Hz) for audio extraction (default: %(default)s)",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
    )
    args = parser.parse_args()

    desired_langs = None
    if args.lang:
        desired_langs = [normalize_language_code(lang) for lang in args.lang.split(",") if lang.strip()]

    print("Extracting audio stream info for file1 …")
    tracks1 = get_audio_tracks(args.file1)

    print("Extracting audio stream info for file2 …")
    tracks2 = get_audio_tracks(args.file2)

    track1, track2 = select_audio_tracks(args.file1, args.file2, tracks1, tracks2, desired_langs)
    print(
        "  Selected file1 audio track: "
        f"index {track1['index']}, codec {track1['codec']}, language {track1['language']}, "
        f"commercial {track1['commercial_name']}"
    )
    print(
        "  Selected file2 audio track: "
        f"index {track2['index']}, codec {track2['codec']}, language {track2['language']}, "
        f"commercial {track2['commercial_name']}"
    )

    # Get overall file durations as fallback for track durations.
    file1_duration = get_file_duration(args.file1)

    # Compute the segment start time for the first track (middle segment of length args.seg_length)
    seg_start = compute_segment_start(track1, file1_duration, args.seg_length)
    print(f"\nFile1: extracting {args.seg_length:.0f} sec segment starting at {seg_start:.2f} s")

    # Extract the segments using ffmpeg
    sr1, audio1 = extract_audio_segment(args.file1, track1["index"], seg_start, args.seg_length, args.sr)
    sr2, audio2 = extract_audio_segment(args.file2, track2["index"], seg_start, args.seg_length, args.sr)
    if sr1 != sr2:
        print("Sample rates do not match. Aborting.")
        sys.exit(1)
    print(f"Extracted audio segments at {sr1} Hz with lengths: file1={len(audio1)} samples, file2={len(audio2)} samples")

    # Compute cross-correlation to find the offset between file2 and file1
    delay_ms, best_lag, max_corr = compute_audio_offset(audio1, audio2, sr1)
    print("\n=== Cross-correlation results ===")
    print(f"Best lag (in samples): {best_lag}")
    print(f"Maximum correlation value: {max_corr:.2f}")
    print(f"Computed sync delay to apply to file2's audio: {delay_ms:.0f} ms")
    print("\nIn mkvtoolnix (mkvmerge), you would use:")
    print(f"    --sync <trackID>:{int(delay_ms)}")
    print("(A negative delay means file2's audio should be advanced relative to file1.)")


if __name__ == "__main__":
    main()
