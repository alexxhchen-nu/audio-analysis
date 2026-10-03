# Audio Analysis

The canonical, reusable audio-analysis workflow is [`notebooks/master.ipynb`](notebooks/master.ipynb). It accepts locally supplied video or audio files (no downloader), extracts video audio with FFmpeg, and includes Demucs stem separation, audio summaries, quality checks, a timestamped manifest, windowed handcrafted features, music analysis, and optional MERT/Basic Pitch sections.

Use [`notebooks/test.ipynb`](notebooks/test.ipynb) as a clean scratch notebook for future sample-specific experiments. Keep generated media and analysis outputs under `video/` and `results/`; source media is ignored by Git.

[`notebooks/notes.ipynb`](notebooks/notes.ipynb) holds methodology notes and the work journal.
