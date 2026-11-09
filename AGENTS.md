# AGENTS

audio-offset-finder estimates the audio offset between two video files with FFmpeg and cross-correlation or FFT, for muxing with `mkvmerge --sync`. README.md has the usage.

## Checks

Run these before committing, with FFmpeg and Python 3.12 or newer installed. CI runs the same steps (`.github/workflows/ci.yml`):

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
examples/make-example.sh
python3 crosscorrelation.py examples/reference.mkv examples/delayed.mkv
python3 fft.py examples/reference.mkv examples/delayed.mkv
```

`examples/make-example.sh` sets the example delay, and `ci.yml` checks the output both scripts print for it.

## Contributing for Someone Else

martinfrancois's rule for outside contributions, and for any agent working for an outside contributor: contributors shouldn't spend time on pull requests that collide with someone else's work or that he wouldn't accept. So a contribution takes one of three paths (see "Three ways to contribute" in CONTRIBUTING.md):

- Small fix (bug fix, typo, docs improvement or similar): open a pull request directly, no issue needed. If an issue for it exists, claim it first as below.
- Existing issue: check its assignees and linked pull requests, post `/take` on the issue, and start only once the assignment comes back. Never open a pull request for an issue that isn't assigned to the contributor you work for, and don't start on one that someone else holds or that an open pull request already addresses.
- New feature: open a thread in the Ideas category of GitHub Discussions. Wait until martinfrancois approves it and opens an issue, then claim that issue with `/take`. Never open a feature pull request without an approved issue assigned to your contributor.
