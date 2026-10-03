# Audio Analysis

The canonical, reusable audio-analysis workflow is [`notebooks/master.ipynb`](notebooks/master.ipynb). It accepts locally supplied video or audio files (no downloader), extracts video audio with FFmpeg, and includes Demucs stem separation, audio summaries, quality checks, a timestamped manifest, windowed handcrafted features, music analysis, and optional MERT/Basic Pitch sections.

## Local media and persistent storage

Place source videos/audio in the existing `video/` folder and set `INPUT_MEDIA` in the master notebook to the filename. Source media in `video/` is excluded from Git; keep generated analysis outputs under `results/`.

For data that must survive remote-machine/container replacement, attach or mount a persistent volume and point the notebook at it by setting `MEDIA_SOURCE_DIR` to the mounted directory before starting the notebook kernel. The default `video/` directory lives inside the project filesystem and is not a backup. The current environment reports an `overlay` filesystem, so persistence across resets/rebuilds is not guaranteed. For durable archives or large collections, use the remote provider's object storage or persistent disk and keep a second copy of irreplaceable originals.

Use [`notebooks/test.ipynb`](notebooks/test.ipynb) as a clean scratch notebook for future sample-specific experiments.

[`notebooks/notes.ipynb`](notebooks/notes.ipynb) holds methodology notes and the work journal.
