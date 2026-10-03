#!/usr/bin/env python3
"""Run a track-grouped DEAM valence/arousal pilot with MERT and Librosa."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, Wav2Vec2FeatureExtractor


ROOT = Path(__file__).resolve().parents[1]
DEAM_ROOT = ROOT / "data" / "emotion_reference" / "DEAM"
AUDIO_ROOT = DEAM_ROOT / "audio" / "MEMD_audio"
ANNOTATION_ROOT = (
    DEAM_ROOT
    / "annotations"
    / "annotations"
    / "annotations averaged per song"
    / "dynamic (per second annotations)"
)
MODEL_ID = "m-a-p/MERT-v1-95M"
WINDOW_SECONDS = 5.0
FIRST_RATING_TIME_SECONDS = 15.0
RATING_STEP_SECONDS = 0.5
MIN_RATINGS_PER_WINDOW = 9
TARGETS = ("valence", "arousal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dynamic_labels() -> tuple[dict[str, dict[float, float]], dict[str, dict[float, float]]]:
    result: dict[str, dict[str, dict[float, float]]] = {}
    for target in TARGETS:
        path = ANNOTATION_ROOT / f"{target}.csv"
        frame = pd.read_csv(path)
        if "song_id" not in frame.columns:
            raise ValueError(f"Missing song_id column in {path}")
        columns: list[tuple[str, float]] = []
        for column in frame.columns:
            match = re.fullmatch(r"sample_(\d+)ms", str(column))
            if match:
                columns.append((column, int(match.group(1)) / 1000.0))
        if not columns:
            raise ValueError(f"No timestamped ratings found in {path}")
        label_map: dict[str, dict[float, float]] = {}
        for row in frame.to_dict(orient="records"):
            song_id = str(row["song_id"])
            label_map[song_id] = {
                timestamp: float(row[column])
                for column, timestamp in columns
                if pd.notna(row[column])
            }
        result[target] = label_map
    return result["valence"], result["arousal"]


def rating_windows(
    song_id: str,
    valence: dict[str, dict[float, float]],
    arousal: dict[str, dict[float, float]],
) -> list[dict[str, float]]:
    valence_track = valence.get(song_id, {})
    arousal_track = arousal.get(song_id, {})
    common_times = sorted(set(valence_track) & set(arousal_track))
    if not common_times:
        return []
    last_start = max(common_times) + RATING_STEP_SECONDS - WINDOW_SECONDS
    starts = np.arange(
        FIRST_RATING_TIME_SECONDS,
        last_start + RATING_STEP_SECONDS / 2,
        WINDOW_SECONDS,
    )
    windows: list[dict[str, float]] = []
    for start in starts:
        end = float(start + WINDOW_SECONDS)
        times = [
            timestamp
            for timestamp in common_times
            if start <= timestamp < end
        ]
        if len(times) < MIN_RATINGS_PER_WINDOW:
            continue
        windows.append(
            {
                "start_seconds": float(start),
                "end_seconds": end,
                "valence": float(np.mean([valence_track[t] for t in times])),
                "arousal": float(np.mean([arousal_track[t] for t in times])),
                "rating_points": len(times),
            }
        )
    return windows


def librosa_features(segment: np.ndarray, sample_rate: int) -> dict[str, float]:
    features: dict[str, float] = {}

    def summarize(name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))

    mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
    for index, row in enumerate(mfcc, start=1):
        summarize(f"mfcc_{index:02d}", row)

    chroma = librosa.feature.chroma_stft(y=segment, sr=sample_rate)
    for index, row in enumerate(chroma):
        features[f"chroma_{index:02d}_mean"] = float(np.mean(row))

    summarize("rms", librosa.feature.rms(y=segment)[0])
    summarize(
        "spectral_centroid",
        librosa.feature.spectral_centroid(y=segment, sr=sample_rate)[0],
    )
    summarize(
        "spectral_bandwidth",
        librosa.feature.spectral_bandwidth(y=segment, sr=sample_rate)[0],
    )
    summarize(
        "spectral_rolloff",
        librosa.feature.spectral_rolloff(y=segment, sr=sample_rate)[0],
    )
    summarize("zero_crossing_rate", librosa.feature.zero_crossing_rate(y=segment)[0])
    contrast = librosa.feature.spectral_contrast(y=segment, sr=sample_rate)
    for index, row in enumerate(contrast):
        summarize(f"spectral_contrast_{index:02d}", row)

    if len(features) != 62:
        raise RuntimeError(f"Expected 62 Librosa features, produced {len(features)}")
    if not np.isfinite(list(features.values())).all():
        raise ValueError("Librosa features contain non-finite values")
    return features


def correlation(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
        return None
    return float(np.corrcoef(actual, predicted)[0, 1])


def metrics_for(
    model_name: str,
    target: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    test_frame: pd.DataFrame,
    fold: int | str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    track_means = (
        test_frame.assign(actual=actual, predicted=predicted)
        .groupby("track_id")[["actual", "predicted"]]
        .mean()
    )
    for unit, truth, estimate in [
        ("window", actual, predicted),
        (
            "track_mean",
            track_means["actual"].to_numpy(),
            track_means["predicted"].to_numpy(),
        ),
    ]:
        rows.extend(
            [
                {
                    "model": model_name,
                    "target": target,
                    "fold": fold,
                    "evaluation_unit": unit,
                    "metric": "mae",
                    "value": float(np.mean(np.abs(truth - estimate))),
                    "n": len(truth),
                },
                {
                    "model": model_name,
                    "target": target,
                    "fold": fold,
                    "evaluation_unit": unit,
                    "metric": "rmse",
                    "value": float(np.sqrt(np.mean(np.square(truth - estimate)))),
                    "n": len(truth),
                },
                {
                    "model": model_name,
                    "target": target,
                    "fold": fold,
                    "evaluation_unit": unit,
                    "metric": "pearson_r",
                    "value": correlation(truth, estimate),
                    "n": len(truth),
                },
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tracks", type=int, default=50)
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="Use every DEAM track with usable ratings and audio.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Use track-grouped K-fold cross-validation instead of one holdout split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Allow Transformers to download MERT model/code if not already cached.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "emotion" / "deam_pilot",
    )
    args = parser.parse_args()
    if not args.all_tracks and args.n_tracks < 10:
        raise ValueError("Use at least 10 tracks for a meaningful grouped holdout pilot")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.cv_folds not in (0,) and args.cv_folds < 2:
        raise ValueError("--cv-folds must be 0 or at least 2")
    if not AUDIO_ROOT.is_dir() or not ANNOTATION_ROOT.is_dir():
        raise FileNotFoundError(
            "DEAM assets are missing. See README.md for the ignored data/emotion_reference/ directory."
        )

    valence, arousal = read_dynamic_labels()
    candidate_ids = sorted(
        song_id
        for song_id in set(valence) & set(arousal)
        if (AUDIO_ROOT / f"{song_id}.mp3").is_file()
        and len(rating_windows(song_id, valence, arousal)) >= 3
    )
    rng = random.Random(args.seed)
    if not args.all_tracks:
        rng.shuffle(candidate_ids)
    target_track_count = len(candidate_ids) if args.all_tracks else args.n_tracks
    window_rows: list[dict[str, Any]] = []
    selected_tracks: list[dict[str, Any]] = []
    sample_rate = 24_000

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=not args.download_model,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=not args.download_model,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model_sample_rate = int(processor.sampling_rate)
    if model_sample_rate != sample_rate:
        raise RuntimeError(
            f"Expected MERT sample rate {sample_rate}, got {model_sample_rate}"
        )

    mert_vectors: list[np.ndarray] = []
    pending_segments: list[np.ndarray] = []

    def encode_batch(batch: list[np.ndarray]) -> None:
        inputs = processor(
            batch,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(output, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("MERT output contains no hidden states")
            hidden = hidden_states[-1]
        pooled = hidden.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        mert_vectors.extend(pooled)

    for candidate_index, song_id in enumerate(candidate_ids, start=1):
        path = AUDIO_ROOT / f"{song_id}.mp3"
        audio, source_rate = librosa.load(path, sr=sample_rate, mono=True)
        duration = len(audio) / sample_rate
        track_windows = [
            row
            for row in rating_windows(song_id, valence, arousal)
            if row["end_seconds"] <= duration + 1e-6
        ]
        if len(track_windows) < 3:
            continue
        selected_tracks.append(
            {
                "track_id": song_id,
                "audio_file": path.name,
                "duration_seconds": duration,
                "audio_sha256": sha256_file(path),
            }
        )
        for window in track_windows:
            start = int(round(window["start_seconds"] * sample_rate))
            end = int(round(window["end_seconds"] * sample_rate))
            segment = audio[start:end]
            if len(segment) != int(WINDOW_SECONDS * sample_rate):
                continue
            feature_row: dict[str, Any] = {
                "track_id": song_id,
                **window,
            }
            feature_row.update(librosa_features(segment, sample_rate))
            window_rows.append(feature_row)
            pending_segments.append(segment.astype(np.float32, copy=False))
            if len(pending_segments) >= args.batch_size:
                encode_batch(pending_segments)
                pending_segments.clear()
        if candidate_index % 100 == 0:
            print(
                f"Processed {candidate_index}/{len(candidate_ids)} candidate tracks; "
                f"{len(selected_tracks)} selected, {len(window_rows)} windows.",
                flush=True,
            )
        if len(selected_tracks) >= target_track_count:
            break

    if pending_segments:
        encode_batch(pending_segments)

    if not args.all_tracks and len(selected_tracks) < args.n_tracks:
        raise RuntimeError(
            f"Only found {len(selected_tracks)} usable tracks; requested {args.n_tracks}"
        )
    if len(selected_tracks) < max(10, args.cv_folds):
        raise RuntimeError(
            f"Only found {len(selected_tracks)} usable tracks; not enough for evaluation"
        )
    if len(window_rows) < 30:
        raise RuntimeError(f"Too few usable windows for the pilot: {len(window_rows)}")

    if len(mert_vectors) != len(window_rows):
        raise RuntimeError("MERT vector count does not match feature window count")
    mert_matrix = np.stack(mert_vectors)
    if mert_matrix.shape[1] != 768:
        raise RuntimeError(f"Expected 768-D MERT embeddings, got {mert_matrix.shape}")

    frame = pd.DataFrame(window_rows)
    librosa_columns = [
        column
        for column in frame.columns
        if column not in {"track_id", "start_seconds", "end_seconds", "rating_points"}
        and column not in TARGETS
    ]
    librosa_matrix = frame[librosa_columns].to_numpy(dtype=np.float64)
    mert_matrix = mert_matrix.astype(np.float64)
    feature_sets = {
        "librosa": librosa_matrix,
        "mert": mert_matrix,
        "librosa+mert": np.concatenate((librosa_matrix, mert_matrix), axis=1),
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    if args.cv_folds:
        group_splitter = GroupKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.seed,
        )
        split_indices = list(
            group_splitter.split(frame, groups=frame["track_id"])
        )
        split_method = f"{args.cv_folds}-fold track-grouped cross-validation"
    else:
        group_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=args.seed,
        )
        split_indices = [
            next(group_splitter.split(frame, groups=frame["track_id"]))
        ]
        split_method = "single 80/20 track-grouped holdout"

    fold_assignments: dict[str, int] = {}
    oof_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for fold, (train_indices, test_indices) in enumerate(split_indices, start=1):
        train = frame.iloc[train_indices].reset_index(drop=True)
        test = frame.iloc[test_indices].reset_index(drop=True)
        train_track_counts = train["track_id"].value_counts()
        sample_weight = train["track_id"].map(
            lambda track_id: 1.0 / train_track_counts[track_id]
        ).to_numpy()
        for track_id in test["track_id"].unique():
            if track_id in fold_assignments:
                raise RuntimeError(f"Track {track_id} appears in more than one test fold")
            fold_assignments[str(track_id)] = fold

        for target in TARGETS:
            y_train = train[target].to_numpy(dtype=np.float64)
            y_test = test[target].to_numpy(dtype=np.float64)
            baseline = np.full_like(y_test, np.average(y_train, weights=sample_weight))
            for model_name, prediction in [
                ("train_track_weighted_mean", baseline),
            ]:
                metric_rows.extend(
                    metrics_for(
                        model_name, target, y_test, prediction, test, fold
                    )
                )
                for row_index, actual, predicted in zip(
                    test_indices, y_test, prediction, strict=True
                ):
                    source = frame.iloc[row_index]
                    prediction_rows.append(
                        {
                            "fold": fold,
                            "track_id": source["track_id"],
                            "start_seconds": source["start_seconds"],
                            "end_seconds": source["end_seconds"],
                            "target": target,
                            "model": model_name,
                            "actual": actual,
                            "predicted": float(predicted),
                        }
                    )
                    oof_rows.setdefault((model_name, target), []).append(
                        {
                            "fold": fold,
                            "row_index": row_index,
                            "actual": actual,
                            "predicted": float(predicted),
                        }
                    )
            for model_name, matrix in feature_sets.items():
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                estimator.fit(
                    matrix[train_indices],
                    y_train,
                    ridge__sample_weight=sample_weight,
                )
                prediction = estimator.predict(matrix[test_indices])
                metric_rows.extend(
                    metrics_for(
                        model_name, target, y_test, prediction, test, fold
                    )
                )
                for row_index, actual, predicted in zip(
                    test_indices, y_test, prediction, strict=True
                ):
                    source = frame.iloc[row_index]
                    prediction_rows.append(
                        {
                            "fold": fold,
                            "track_id": source["track_id"],
                            "start_seconds": source["start_seconds"],
                            "end_seconds": source["end_seconds"],
                            "target": target,
                            "model": model_name,
                            "actual": actual,
                            "predicted": float(predicted),
                        }
                    )
                    oof_rows.setdefault((model_name, target), []).append(
                        {
                            "fold": fold,
                            "row_index": row_index,
                            "actual": actual,
                            "predicted": float(predicted),
                        }
                    )

    for (model_name, target), rows in oof_rows.items():
        ordered_rows = sorted(rows, key=lambda row: row["row_index"])
        row_indices = [row["row_index"] for row in ordered_rows]
        evaluation_frame = frame.iloc[row_indices].reset_index(drop=True)
        metric_rows.extend(
            metrics_for(
                model_name,
                target,
                np.asarray([row["actual"] for row in ordered_rows]),
                np.asarray([row["predicted"] for row in ordered_rows]),
                evaluation_frame,
                "pooled_oof",
            )
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_tracks).to_csv(
        output_dir / "selected_tracks.csv", index=False
    )
    frame[
        ["track_id", "start_seconds", "end_seconds", "rating_points", *TARGETS]
        + librosa_columns
    ].to_csv(output_dir / "window_labels_and_librosa.csv", index=False)
    np.save(output_dir / "mert_embeddings.npy", mert_matrix)
    frame[["track_id", "start_seconds", "end_seconds", *TARGETS]].to_csv(
        output_dir / "window_index.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    fold_frame = pd.DataFrame(
        [
            {"track_id": track_id, "test_fold": fold}
            for track_id, fold in sorted(fold_assignments.items())
        ]
    )
    fold_frame.to_csv(output_dir / "track_folds.csv", index=False)
    if len(fold_assignments) != frame["track_id"].nunique():
        raise RuntimeError("Not every track was assigned to exactly one test fold")

    first_train_indices, first_test_indices = split_indices[0]
    train_ids = sorted(frame.iloc[first_train_indices]["track_id"].unique().tolist())
    test_ids = sorted(frame.iloc[first_test_indices]["track_id"].unique().tolist())
    config = {
        "dataset": "DEAM",
        "dataset_audio_sha256": sha256_file(DEAM_ROOT / "DEAM_audio.zip"),
        "annotation_sha256": sha256_file(DEAM_ROOT / "DEAM_Annotations.zip"),
        "model_id": MODEL_ID,
        "sample_rate_hz": sample_rate,
        "window_seconds": WINDOW_SECONDS,
        "window_hop_seconds": WINDOW_SECONDS,
        "first_rating_time_seconds": FIRST_RATING_TIME_SECONDS,
        "rating_step_seconds": RATING_STEP_SECONDS,
        "minimum_rating_points_per_window": MIN_RATINGS_PER_WINDOW,
        "pooling": "last_hidden_state_mean_over_time",
        "librosa_feature_count": len(librosa_columns),
        "mert_dimensions": int(mert_matrix.shape[1]),
        "requested_tracks": None if args.all_tracks else args.n_tracks,
        "all_tracks_requested": args.all_tracks,
        "selected_tracks": len(selected_tracks),
        "windows": len(frame),
        "split_method": split_method,
        "cv_folds": args.cv_folds or None,
        "train_tracks": len(train_ids),
        "test_tracks": len(test_ids),
        "train_windows": len(first_train_indices),
        "test_windows": len(first_test_indices),
        "fold_track_counts": {
            str(fold): int((fold_frame["test_fold"] == fold).sum())
            for fold in sorted(fold_frame["test_fold"].unique())
        },
        "seed": args.seed,
        "ridge_alpha": 10.0,
        "device": str(device),
        "train_track_ids": train_ids,
        "test_track_ids": test_ids,
        "scope_note": (
            "Exploratory evaluation only. All windows from each track are assigned "
            "to one test fold, so there is no track leakage. Non-overlapping windows "
            "are used. DEAM performance does not establish "
            "transfer to project short-video audio."
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selected_tracks": len(selected_tracks),
                "windows": len(frame),
                "split_method": split_method,
                "models": list(feature_sets),
                "train_tracks": len(train_ids),
                "test_tracks": len(test_ids),
                "folds": args.cv_folds or 1,
                "device": str(device),
                "pooled_oof_metrics": [
                    row for row in metric_rows if row["fold"] == "pooled_oof"
                ],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
