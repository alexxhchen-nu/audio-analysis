# 🌱 DEAM Emotion Pilot

This is a small, reproducible test of whether timestamp-aligned audio features can predict DEAM's human valence/arousal ratings on songs held out from training.

## Design

- 50 deterministically sampled DEAM tracks (`seed=42`)
- 5-second, non-overlapping windows at 15, 20, 25, 30, 35, and 40 seconds
- 300 windows total; each window is aligned to 10 half-second DEAM rating points
- 40 training tracks / 240 windows and 10 held-out test tracks / 60 windows
- 62 Librosa features versus 768-dimensional mean-pooled MERT embeddings
- StandardScaler + Ridge (`alpha=10`), with equal total training weight per track
- Training-set track-weighted mean as the baseline

Run from the repository root:

```bash
python scripts/deam_emotion_pilot.py --n-tracks 50 --seed 42
```

## Held-out results

Window-level metrics (lower MAE/RMSE is better; higher Pearson correlation is better):

| Target | Features | MAE | RMSE | Pearson r |
|---|---|---:|---:|---:|
| Valence | Train-track-weighted mean baseline | 0.238 | 0.266 | — |
| Valence | Librosa | **0.160** | **0.207** | 0.605 |
| Valence | MERT | 0.166 | 0.219 | **0.675** |
| Arousal | Train-track-weighted mean baseline | 0.309 | 0.342 | — |
| Arousal | Librosa | **0.194** | 0.251 | 0.710 |
| Arousal | MERT | 0.208 | **0.243** | **0.731** |

The baseline correlation is undefined because it predicts a constant. Track-mean results and all exact values are in [`metrics.csv`](metrics.csv).

## Interpretation

Both representations beat the simple mean baseline on this one held-out split. In this small pilot, Librosa has lower window-level error for both targets; MERT has higher window-level correlation. This does **not** establish that one method is generally better: the test set is only 10 songs, this is one split, and no hyperparameter tuning, feature fusion, or uncertainty analysis was performed.

These results show preliminary predictability of DEAM's dataset-specific human ratings—not a validated music-emotion detector and not evidence that the model can infer a creator's or listener's inner state. They do not demonstrate transfer to this project's short-video audio. Testing that requires human ratings on project clips and a separate held-out evaluation.

## Files

- [`config.json`](config.json): split, windowing, model, and data-hash metadata
- [`selected_tracks.csv`](selected_tracks.csv): sampled DEAM track IDs and audio hashes
- [`window_index.csv`](window_index.csv): track/time-window/label index
- [`window_labels_and_librosa.csv`](window_labels_and_librosa.csv): window labels and handcrafted features
- [`mert_embeddings.npy`](mert_embeddings.npy): MERT window vectors (ignored by Git)
- [`test_predictions.csv`](test_predictions.csv): held-out predictions
- [`metrics.csv`](metrics.csv): window- and track-mean evaluation metrics

The raw DEAM audio and annotation datasets remain outside Git under the ignored `data/emotion_reference/` directory.
