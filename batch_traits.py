#!/usr/bin/env python3
"""扩展特质批处理：为 results/master 下各视频计算视频级扩展音乐特质。

新增特质（均基于 no_vocals 轨）：
- spectral_flatness mean/std  —— 噪声度/现场感
- tonnetz 六维均值 + 轨迹方差 —— 调性位置与转调稳定性
- tempogram_ratio 主比        —— 二拍系 vs 三拍系律动
- HPSS 打击能量比             —— 节奏驱动 vs 旋律驱动
- vocals 能量比               —— 人声占比
- bgm_type 规则分类           —— 音乐化 BGM / 纯人声解说 / 纯打击 / 环境声

增量模式：audio_traits_extended.csv 中已有且 no_vocals.wav mtime 未变的视频跳过。
6 进程并行。产物：results/master/audio_traits_extended.csv（每视频一行）。
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

PROJECT_ROOT = Path("/home/ubuntu/music")
RESULTS_ROOT = PROJECT_ROOT / "results" / "master"
OUT_CSV = RESULTS_ROOT / "audio_traits_extended.csv"
WORKERS = 6
SR = 22050

COLS = [
    "account_id", "video_id", "duration_s",
    "flatness_mean", "flatness_std",
    "tonnetz_1", "tonnetz_2", "tonnetz_3", "tonnetz_4", "tonnetz_5", "tonnetz_6",
    "tonnetz_traj_var",
    "tempo_bpm", "tempogram_ratio",
    "percussive_ratio", "vocals_ratio",
    "bgm_type",
    "stem_mtime",
]


def rms_energy(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y ** 2)) + 1e-12)


def classify_bgm(vocals_ratio: float, flat: float, perc_ratio: float) -> str:
    """规则分类，阈值基于分布校准（见 notes.ipynb bgm_type 节）。"""
    if vocals_ratio > 0.7 and flat > 0.15:
        return "speech_dominant"
    if vocals_ratio <= 0.7 and perc_ratio > 0.6:
        return "percussion_dominant"
    if vocals_ratio <= 0.7 and flat > 0.4:
        return "ambient_field"
    return "musical_bgm"


def process_one(args) -> dict | None:
    account_id, video_id, video_dir = args
    video_dir = Path(video_dir)
    stem_dir = video_dir / "demucs" / "htdemucs" / "audio_original"
    no_vocals = stem_dir / "no_vocals.wav"
    vocals = stem_dir / "vocals.wav"
    if not no_vocals.is_file():
        return None
    try:
        y, sr = librosa.load(no_vocals, sr=SR, mono=True)
        duration = len(y) / sr
        if duration < 3:
            return None

        flat = librosa.feature.spectral_flatness(y=y)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)          # (6, n_frames)
        # 调性轨迹方差：六维坐标随时间的平均欧氏步长
        steps = np.linalg.norm(np.diff(tonnetz, axis=1), axis=0)
        traj_var = float(np.mean(steps)) if len(steps) else 0.0

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        try:
            from librosa.feature.rhythm import tempo as _rhythm_tempo
            tempo = float(_rhythm_tempo(onset_envelope=onset_env, sr=sr)[0])
        except ImportError:
            tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0])
        tg = librosa.feature.tempogram_ratio(onset_envelope=onset_env, sr=sr)
        tempo_ratio = float(tg.mean(axis=1)[1]) if tg.shape[0] > 1 else 0.0

        harm, perc = librosa.effects.hpss(y)
        perc_ratio = rms_energy(perc) / (rms_energy(harm) + rms_energy(perc))

        vocals_ratio = 0.0
        if vocals.is_file():
            yv, _ = librosa.load(vocals, sr=SR, mono=True)
            n = min(len(y), len(yv))
            vocals_ratio = rms_energy(yv[:n]) / (rms_energy(y[:n]) + rms_energy(yv[:n]))

        flat_mean = float(flat.mean())
        row = {
            "account_id": account_id,
            "video_id": video_id,
            "duration_s": round(duration, 2),
            "flatness_mean": round(flat_mean, 4),
            "flatness_std": round(float(flat.std()), 4),
            **{f"tonnetz_{i+1}": round(float(v), 4) for i, v in enumerate(tonnetz.mean(axis=1))},
            "tonnetz_traj_var": round(traj_var, 4),
            "tempo_bpm": round(tempo, 1),
            "tempogram_ratio": round(tempo_ratio, 3),
            "percussive_ratio": round(perc_ratio, 4),
            "vocals_ratio": round(vocals_ratio, 4),
            "bgm_type": classify_bgm(vocals_ratio, flat_mean, perc_ratio),
            "stem_mtime": no_vocals.stat().st_mtime,
        }
        return row
    except Exception as exc:
        print(f"FAIL {account_id}/{video_id}: {exc}", flush=True)
        return None


def main() -> None:
    existing = pd.DataFrame(columns=COLS)
    if OUT_CSV.is_file():
        existing = pd.read_csv(OUT_CSV)
    done = {}
    if len(existing):
        done = {r.video_id: r.stem_mtime for r in existing.itertuples()}

    tasks = []
    for account_dir in sorted(RESULTS_ROOT.iterdir()):
        if not account_dir.is_dir():
            continue
        for video_dir in sorted(account_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            stem = video_dir / "demucs" / "htdemucs" / "audio_original" / "no_vocals.wav"
            if not stem.is_file():
                continue
            if done.get(video_dir.name) == stem.stat().st_mtime:
                continue
            tasks.append((account_dir.name, video_dir.name, str(video_dir)))

    print(f"待处理 {len(tasks)} 个视频（已完成 {len(done)}）", flush=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if row:
                rows.append(row)
            if i % 20 == 0:
                print(f"{i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    if rows:
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out = out[COLS]
        out.to_csv(OUT_CSV, index=False)
        print(f"写入 {len(rows)} 行 -> {OUT_CSV}（总计 {len(out)} 行）", flush=True)
    else:
        print("无新增", flush=True)


if __name__ == "__main__":
    main()
