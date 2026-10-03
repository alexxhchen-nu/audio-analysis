# ⚡ DEAM Full-Corpus Cross-Validation

This exploratory benchmark tests whether aligned Librosa features and MERT embeddings predict DEAM's human valence/arousal ratings on songs excluded from each model's training fold.

## Evaluation design

- **1,802 tracks**, each assigned to exactly one of five shuffled, track-grouped test folds (`seed=42`)
- **12,964 non-overlapping 5-second windows**, aligned to DEAM's ratings from 15 seconds onward
- Each window label averages its 10 corresponding 500-ms valence/arousal ratings
- **62 Librosa features**, **768-D mean-pooled MERT embeddings**, and their concatenation
- StandardScaler + Ridge (`alpha=10`), with equal total training weight per track
- Training-fold track-weighted mean baseline
- Hyperparameters fixed from the pilot; no tuning or feature selection

The feature and label extraction run:

```bash
python scripts/deam_emotion_pilot.py \
  --all-tracks --cv-folds 5 --seed 42 \
  --output-dir results/emotion/deam_cross_validation
```

To rerun only the grouped evaluation from already cached features:

```bash
python scripts/deam_emotion_cv.py \
  --features-dir results/emotion/deam_cross_validation \
  --folds 5 --seed 42
```

## Pooled out-of-fold results

Each prediction is from a model that did not train on that song. Lower MAE/RMSE is better; higher Pearson correlation is better.

### Per-window metrics

| Target | Features | MAE | RMSE | Pearson r |
|---|---|---:|---:|---:|
| Valence | Track-weighted mean baseline | 0.205 | 0.252 | -0.073 |
| Valence | Librosa | 0.180 | 0.232 | 0.395 |
| Valence | MERT | 0.167 | 0.217 | 0.528 |
| Valence | Librosa + MERT | **0.164** | **0.214** | **0.547** |
| Arousal | Track-weighted mean baseline | 0.236 | 0.282 | -0.047 |
| Arousal | Librosa | 0.154 | 0.193 | 0.729 |
| Arousal | MERT | 0.149 | 0.190 | 0.743 |
| Arousal | Librosa + MERT | **0.143** | **0.182** | **0.766** |

### Per-track mean metrics

| Target | Features | MAE | RMSE | Pearson r |
|---|---|---:|---:|---:|
| Valence | Track-weighted mean baseline | 0.194 | 0.235 | -0.076 |
| Valence | Librosa | 0.158 | 0.198 | 0.536 |
| Valence | MERT | 0.138 | 0.173 | 0.674 |
| Valence | Librosa + MERT | **0.135** | **0.171** | **0.685** |
| Arousal | Track-weighted mean baseline | 0.235 | 0.280 | -0.052 |
| Arousal | Librosa | 0.138 | 0.174 | 0.784 |
| Arousal | MERT | 0.126 | 0.162 | 0.815 |
| Arousal | Librosa + MERT | **0.119** | **0.154** | **0.835** |

## Interpretation and limits

On this DEAM benchmark, combining the feature groups gave the best pooled scores for both targets. MERT alone also outperformed Librosa alone on the pooled metrics here. These are dataset-specific prediction results, not proof that each feature group independently represents a named emotion or that the combined model understands emotion.

The evaluation uses the full DEAM corpus and prevents song-level leakage, but it is still one fixed five-fold partition with one fixed Ridge setting. It does not test transfer to short-video audio, Chinese-language clips, Demucs stems, or this project's sources. Validate transfer with independent human ratings on project clips before making claims or using model scores as labels. Ratings describe the dataset's annotation task, not a listener's or creator's hidden mental state.

## Output files

- [`config.json`](config.json): source hashes, preprocessing, features, and fold metadata
- [`selected_tracks.csv`](selected_tracks.csv): included track IDs and source audio hashes
- [`track_folds.csv`](track_folds.csv): one held-out fold per song
- [`window_index.csv`](window_index.csv): window IDs and aligned labels
- [`window_labels_and_librosa.csv`](window_labels_and_librosa.csv): labels and handcrafted features
- [`mert_embeddings.npy`](mert_embeddings.npy): MERT vectors (ignored by Git)
- [`test_predictions.csv`](test_predictions.csv): all out-of-fold predictions
- [`metrics.csv`](metrics.csv): per-fold and pooled out-of-fold metrics

Raw DEAM data remain outside Git in the ignored `data/emotion_reference/` directory.
