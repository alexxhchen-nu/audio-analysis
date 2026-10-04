#!/usr/bin/env python3
"""多账号批量分析：依次对 video/ 下 6 个新账号跑 master.ipynb 单视频管线。

每账号内部 6 路并行（Demucs/MERT 主要吃单核 CPU + 少量 GPU），账号间串行，
避免跨账号争抢资源。断点续跑：已有 MERT 索引的视频跳过。
全部账号完成后统一 Milvus 增量入库（单进程，避免并行写 Lite 库）。"""
import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORKERS = 6
ACCOUNTS = ["半月谈", "光明日报", "经济日报", "央视新闻", "人民陆军", "南海舰队"]

PROJECT_ROOT = Path("/home/ubuntu/music")
VIDEO_ROOT = PROJECT_ROOT / "video"
RUN_ROOT = Path("/tmp/master_runs_multi")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
MEDIA_ROOT = Path("/tmp/master_media")
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

# 从 master.ipynb 生成批处理副本（语料级 cell 不跑）
KEEP_CODE_CELLS = {4, 6, 7, 9, 11, 13, 15, 17, 19}
nb = json.load(open(PROJECT_ROOT / "notebooks" / "master.ipynb"))
cells = [c for idx, c in enumerate(nb["cells"])
         if c["cell_type"] == "markdown" or idx in KEEP_CODE_CELLS]
batch_path = PROJECT_ROOT / "notebooks" / "master_batch.ipynb"
json.dump(dict(nb, cells=cells), open(batch_path, "w"), ensure_ascii=False, indent=1)

_log_lock = threading.Lock()

def make_log(account):
    log_path = RUN_ROOT / f"batch_{account}.log"
    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] [{account}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
    return log

all_failures = {}
t_all = time.time()

for account in ACCOUNTS:
    log = make_log(account)
    video_dir = VIDEO_ROOT / account
    results_root = PROJECT_ROOT / "results" / "master" / account
    run_dir = RUN_ROOT / account
    run_dir.mkdir(exist_ok=True)

    link = MEDIA_ROOT / account
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(video_dir)

    videos = sorted(p.name for p in video_dir.iterdir() if p.suffix.lower() == ".mp4")
    total = len(videos)

    def is_done(name):
        return (results_root / Path(name).stem / "mert" / "mert_v1_95m_embedding_index.csv").is_file()

    pending = [(i, name) for i, name in enumerate(videos) if not is_done(name)]
    log(f"共 {total} 个视频，已完成 {total - len(pending)}，待处理 {len(pending)}，并行度 {WORKERS}")

    env_base = dict(os.environ, ACCOUNT_ID=account, MEDIA_SOURCE_DIR=str(MEDIA_ROOT))

    def run_one(item):
        i, name = item
        out_nb = run_dir / f"run_{i:03d}.ipynb"
        env = dict(env_base, INPUT_FILENAME=name)
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook",
             "--execute", str(batch_path), "--output", str(out_nb),
             "--ExecutePreprocessor.timeout=3600"],
            cwd=str(PROJECT_ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        ok = proc.returncode == 0
        with _log_lock:
            log(f"{'OK  ' if ok else 'FAIL'} {i+1}/{total} ({time.time()-t0:.0f}s) {name}")
        return None if ok else name

    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, item) for item in pending]
        for future in as_completed(futures):
            r = future.result()
            if r:
                failures.append(r)

    log(f"账号完成: {total - len(failures)} ok, {len(failures)} failed")
    for name in failures:
        log(f"failed: {name}")
    all_failures[account] = failures

log = make_log("ALL")
log(f"全部账号完成，耗时 {(time.time()-t_all)/3600:.1f}h")
for acc, fails in all_failures.items():
    log(f"{acc}: {len(fails)} failed")

# 统一 Milvus 增量入库
try:
    sys.path.insert(0, str(PROJECT_ROOT))
    from milvus_store import ingest_new
    added, skipped = ingest_new()
    log(f"Milvus 增量入库: {added} 新增/更新, {skipped} 跳过")
except Exception as exc:
    log(f"Milvus 增量入库失败（可手动运行 python3 milvus_store.py 补入）: {exc}")

sys.exit(1 if any(all_failures.values()) else 0)
