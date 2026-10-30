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


@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_ffmpeg_segment_has_exact_samples_without_wav_warnings(tmp_path, sample_rate):
    import warnings
    from crosscorrelation import extract_audio_segment

    source = tmp_path / "source.wav"
    samples = to_int16(make_noise(sample_rate, 2))
    wavfile.write(source, sample_rate, samples)

    with warnings.catch_warnings():
        warnings.simplefilter("error", wavfile.WavFileWarning)
        rate, actual = extract_audio_segment(
            str(source), 0, 0.25, seg_length=1, sample_rate=sample_rate
        )

    assert rate == sample_rate
    assert len(actual) == sample_rate
    np.testing.assert_array_equal(actual, samples[sample_rate // 4:sample_rate * 5 // 4])


@pytest.mark.parametrize("malformed", [False, True], ids=["truncated", "malformed"])
def test_ffmpeg_rejects_invalid_audio(tmp_path, malformed, capsys):
    from crosscorrelation import extract_audio_segment

    source = tmp_path / "broken.wav"
    if malformed:
        source.write_bytes(b"not a WAV file")
    else:
        wavfile.write(source, 8000, to_int16(make_noise(8000, 1)))
        source.write_bytes(source.read_bytes()[:8044])

    with pytest.raises(SystemExit) as error:
        extract_audio_segment(str(source), 0, 0, seg_length=1, sample_rate=8000)

    assert error.value.code == 1
    output = capsys.readouterr()
    assert str(source) in output.out + output.err


def test_fft_alignment_keeps_equal_basenames_separate(tmp_path, monkeypatch):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    source = to_int16(make_noise(48000, 3))
    delayed = shift(source, 5120).astype(np.int16)
    wavfile.write(first / "audio.wav", 48000, source)
    wavfile.write(second / "audio.wav", 48000, delayed)
    sentinel = tmp_path / "audio_middle.wav"
    sentinel.write_bytes(b"existing user file")
    monkeypatch.chdir(tmp_path)

    actual = fft.align(str(first / "audio.wav"), str(second / "audio.wav"))

    assert actual == pytest.approx(-5120 / 48000)
    assert sentinel.read_bytes() == b"existing user file"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audio_middle.wav", "one", "two"]


def test_fft_extraction_failure_removes_temporary_outputs(tmp_path, monkeypatch):
    from pathlib import Path

    created = []

    def failing_extract(_input, _start, _length, output):
        path = Path(output)
        path.write_bytes(b"partial audio")
        created.append(path)
        raise SystemExit(1)

    monkeypatch.setattr(fft, "get_duration", lambda _path: 1)
    monkeypatch.setattr(fft, "extract_audio", failing_extract)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        fft.align("one/audio.wav", "two/audio.wav")

    assert created
    assert not created[0].exists()
    assert not list(tmp_path.iterdir())
