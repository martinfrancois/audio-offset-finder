import json

import pytest

import crosscorrelation


def test_language_aliases_and_titles_pair_the_matching_audio_tracks(monkeypatch):
    streams = {
        "reference.mkv": [
            {"index": 1, "codec_name": "dts", "tags": {"LANGUAGE": "deu", "TITLE": "German DTS-HD (5.1)"}},
            {"index": 2, "codec_name": "aac", "tags": {"language": "eng", "title": "English AAC"}},
        ],
        "other.mkv": [
            {"index": 4, "codec_name": "aac", "tags": {"language": "eng", "title": "English AAC"}},
            {"index": 5, "codec_name": "dts", "tags": {"language": "ger", "title": "Deutsch DTS-HD (5.1)"}},
        ],
    }
    monkeypatch.setattr(
        crosscorrelation, "run_ffprobe", lambda command: json.dumps({"streams": streams[command[-1]]})
    )
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("An unambiguous pair must not prompt"))

    first, second = crosscorrelation.select_audio_tracks(
        "reference.mkv", "other.mkv",
        crosscorrelation.get_audio_tracks("reference.mkv"),
        crosscorrelation.get_audio_tracks("other.mkv"),
        ["de"],
    )

    assert (first["index"], second["index"]) == (1, 5)
    assert first["commercial_key"] == second["commercial_key"] == "dts-hd"


def test_ambiguous_codec_groups_use_one_selection_for_both_files(monkeypatch):
    def track(index, codec):
        return {
            "index": index, "codec": codec, "language": "eng", "title": codec,
            "commercial_name": codec, "commercial_key": codec.casefold(),
        }

    prompts = []

    def choose(prompt):
        prompts.append(prompt)
        return "2"

    monkeypatch.setattr("builtins.input", choose)

    first, second = crosscorrelation.select_audio_tracks(
        "reference.mkv", "other.mkv",
        [track(1, "AAC"), track(2, "DTS")],
        [track(4, "DTS"), track(5, "AAC")],
        ["en"],
    )

    assert (first["index"], second["index"]) == (2, 4)
    assert len(prompts) == 1


def test_missing_requested_language_stops_before_prompting(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Missing language must not prompt"))
    tracks = [{"index": 1, "codec": "aac", "language": "eng"}]

    with pytest.raises(SystemExit) as error:
        crosscorrelation.select_audio_tracks("reference.mkv", "other.mkv", tracks, tracks, ["de"])

    assert error.value.code == 1
    assert "Available languages: eng" in capsys.readouterr().out
