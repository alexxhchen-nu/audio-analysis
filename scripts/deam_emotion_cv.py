#!/usr/bin/env python3
"""Evaluate cached DEAM window features with track-grouped cross-validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deam_emotion_pilot import ROOT, TARGETS, metrics_for


DEFAULT_FEATURE_DIR = ROOT / "results" / "emotion" / "deam_cross_validation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.ridge_alpha <= 0:
        raise ValueError("--ridge-alpha must be positive")

    source_dir = args.features_dir.resolve()
    required_files = (
        "window_labels_and_librosa.csv",
        "mert_embeddings.npy",
        "selected_tracks.csv",
        "config.json",
    )
    for filename in required_files:
        if not (source_dir / filename).is_file():
            raise FileNotFoundError(f"Missing cached feature file: {source_dir / filename}")

    frame = pd.read_csv(source_dir / "window_labels_and_librosa.csv")
    mert_matrix = np.load(source_dir / "mert_embeddings.npy", mmap_mode="r")
    if mert_matrix.ndim != 2 or mert_matrix.shape != (len(frame), 768):
        raise ValueError(
            f"Expected MERT matrix shape ({len(frame)}, 768), got {mert_matrix.shape}"
        )
    if frame["track_id"].nunique() < args.folds:
        raise ValueError("Number of unique tracks is smaller than requested CV folds")
    if not np.isfinite(mert_matrix).all():
        raise ValueError("MERT matrix contains non-finite values")

    excluded = {"track_id", "start_seconds", "end_seconds", "rating_points", *TARGETS}
    librosa_columns = [column for column in frame.columns if column not in excluded]
    if len(librosa_columns) != 62:
        raise ValueError(f"Expected 62 Librosa features, found {len(librosa_columns)}")
    librosa_matrix = frame[librosa_columns].to_numpy(dtype=np.float64)
    mert_matrix = np.asarray(mert_matrix, dtype=np.float64)
    feature_sets = {
        "librosa": librosa_matrix,
        "mert": mert_matrix,
        "librosa+mert": np.concatenate((librosa_matrix, mert_matrix), axis=1),
    }

    splitter = GroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_by_track: dict[str, int] = {}
    oof_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(frame, groups=frame["track_id"]),
        start=1,
    ):
        train = frame.iloc[train_indices].reset_index(drop=True)
        test = frame.iloc[test_indices].reset_index(drop=True)
        train_counts = train["track_id"].value_counts()
        sample_weight = train["track_id"].map(
            lambda track_id: 1.0 / train_counts[track_id]
        ).to_numpy()
        for track_id in test["track_id"].unique():
            track_id = str(track_id)
            if track_id in fold_by_track:
                raise RuntimeError(f"Track {track_id} occurs in multiple test folds")
            fold_by_track[track_id] = fold

        for target in TARGETS:
            y_train = train[target].to_numpy(dtype=np.float64)
            y_test = test[target].to_numpy(dtype=np.float64)
            baseline = np.full_like(y_test, np.average(y_train, weights=sample_weight))
            predictions = {
                "train_track_weighted_mean": baseline,
            }
            for model_name, matrix in feature_sets.items():
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=args.ridge_alpha),
                )
                model.fit(
                    matrix[train_indices],
                    y_train,
                    ridge__sample_weight=sample_weight,
                )
                predictions[model_name] = model.predict(matrix[test_indices])

            for model_name, predicted in predictions.items():
                metric_rows.extend(
                    metrics_for(
                        model_name,
                        target,
                        y_test,
                        predicted,
                        test,
                        fold,
                    )
                )
                for row_index, actual, estimate in zip(
                    test_indices,
                    y_test,
                    predicted,
                    strict=True,
                ):
                    source = frame.iloc[row_index]
                    prediction_rows.append(
                        {
                            "fold": fold,
                            "track_id": str(source["track_id"]),
                            "start_seconds": float(source["start_seconds"]),
                            "end_seconds": float(source["end_seconds"]),
                            "target": target,
                            "model": model_name,
                            "actual": float(actual),
                            "predicted": float(estimate),
                        }
                    )
                    oof_rows.setdefault((model_name, target), []).append(
                        {
                            "row_index": int(row_index),
                            "actual": float(actual),
                            "predicted": float(estimate),
                        }
                    )
        print(
            f"Completed fold {fold}/{args.folds}: "
            f"{train['track_id'].nunique()} train tracks, "
            f"{test['track_id'].nunique()} held-out tracks.",
            flush=True,
        )

    for (model_name, target), rows in oof_rows.items():
        ordered_rows = sorted(rows, key=lambda row: row["row_index"])
        row_indices = [row["row_index"] for row in ordered_rows]
        metric_rows.extend(
            metrics_for(
                model_name,
                target,
                np.asarray([row["actual"] for row in ordered_rows]),
                np.asarray([row["predicted"] for row in ordered_rows]),
                frame.iloc[row_indices].reset_index(drop=True),
                "pooled_oof",
            )
        )

    if len(fold_by_track) != frame["track_id"].nunique():
        raise RuntimeError("Not every track was assigned to exactly one test fold")

    output_dir = (args.output_dir or source_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    fold_frame = pd.DataFrame(
        [
            {"track_id": track_id, "test_fold": fold}
            for track_id, fold in sorted(fold_by_track.items())
        ]
    )
    fold_frame.to_csv(output_dir / "track_folds.csv", index=False)

    config_path = source_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "requested_tracks": None,
            "all_tracks_requested": True,
            "selected_tracks": int(frame["track_id"].nunique()),
            "windows": len(frame),
            "split_method": f"{args.folds}-fold shuffled track-grouped cross-validation",
            "cv_folds": args.folds,
            "seed": args.seed,
            "ridge_alpha": args.ridge_alpha,
            "fold_track_counts": {
                str(fold): int((fold_frame["test_fold"] == fold).sum())
                for fold in range(1, args.folds + 1)
            },
            "scope_note": (
                "Exploratory evaluation only. All windows from each track are "
                "assigned to exactly one test fold; windows do not overlap. "
                "DEAM results do not establish transfer to project short-video audio."
            ),
        }
    )
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pooled = [
        row for row in metric_rows if row["fold"] == "pooled_oof"
    ]
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "tracks": int(frame["track_id"].nunique()),
                "windows": len(frame),
                "folds": args.folds,
                "pooled_oof_metrics": pooled,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
