import numpy as np
import pytest
from scipy.io import wavfile

import fft
from crosscorrelation import compute_audio_offset


def make_noise(sample_rate, seconds, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.3, sample_rate * seconds)


def shift(signal, samples):
    """Delay the signal by `samples` (advance it when negative), keeping the length."""
    if samples >= 0:
        return np.concatenate([np.zeros(samples), signal])[: len(signal)]
    return np.concatenate([signal[-samples:], np.zeros(-samples)])


def to_int16(signal):
    return (signal * 32767).astype(np.int16)


def write_and_read_wav(path, sample_rate, signal):
    wavfile.write(path, sample_rate, to_int16(signal))
    rate, data = wavfile.read(path)
    return rate, data


@pytest.mark.parametrize("delay_ms", [250, -120, 0])
def test_cross_correlation_recovers_known_offset(tmp_path, delay_ms):
    sample_rate = 8000
    reference = make_noise(sample_rate, 3)
    other = shift(reference, delay_ms * sample_rate // 1000)
    rate1, seg1 = write_and_read_wav(tmp_path / "reference.wav", sample_rate, reference)
    rate2, seg2 = write_and_read_wav(tmp_path / "other.wav", sample_rate, other)
    assert rate1 == rate2 == sample_rate

    estimated_ms, best_lag, _ = compute_audio_offset(seg1, seg2, sample_rate)

    # A file whose audio arrives later must be advanced, so the sign flips.
    assert estimated_ms == pytest.approx(-delay_ms, abs=1.0)
    assert best_lag == delay_ms * sample_rate // 1000


@pytest.mark.parametrize("delay_bins", [23, -10])
def test_fft_compute_delay_recovers_known_offset(delay_bins):
    sample_rate = fft.SAMPLE_RATE
    fft_bin_size = fft.DEFAULT_FFT_BIN_SIZE
    seconds = 20
    t = np.arange(sample_rate * seconds) / sample_rate
    # Noise plus two tones that switch on and off, so the spectrum changes over time.
    reference = make_noise(sample_rate, seconds)
    reference += 0.3 * np.sin(2 * np.pi * 440 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    reference += 0.3 * np.sin(2 * np.pi * 1320 * t) * (np.sin(2 * np.pi * 0.7 * t + 1) > 0)
    other = shift(reference, delay_bins * fft_bin_size)

    estimated_seconds = fft.compute_delay(to_int16(reference), to_int16(other), sample_rate)

    # A file whose audio arrives later must be advanced, so the sign flips.
    assert estimated_seconds == pytest.approx(-delay_bins * fft_bin_size / sample_rate, abs=1e-9)


def test_ffprobe_reads_audio_language_tag(tmp_path):
    import subprocess
    from crosscorrelation import get_audio_tracks

    sample = tmp_path / "language.mka"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono",
        "-t", "0.1", "-c:a", "pcm_s16le", "-metadata:s:a:0", "language=eng", str(sample),
    ], check=True)
    tracks = get_audio_tracks(str(sample))
    assert len(tracks) == 1
    assert tracks[0]["language"] == "eng"
