"""Milvus Lite 向量存储：导入 results/master 下各视频的 MERT 向量，支持相似度检索。

用法:
    from milvus_store import ingest_new, upsert_video, search_similar
    ingest_new()                       # 增量导入：只处理新增/有更新的视频
    upsert_video(results_dir)          # 单视频入库（master.ipynb 工作流末尾自动调用）
    hits = search_similar("人民日报", "<视频名>")  # 以某视频为查询，找最近邻
"""
import os
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "master"
DB_PATH = PROJECT_ROOT / "results" / "milvus_mert.db"
COLLECTION = "mert_embeddings"
DIM = 768  # MERT-v1-95M 输出维度

# 默认 Milvus Lite（本地文件）；设 MILVUS_URI=http://127.0.0.1:19530 走 Server
MILVUS_URI = os.environ.get("MILVUS_URI", str(DB_PATH))


def get_client() -> MilvusClient:
    client = MilvusClient(MILVUS_URI)
    _ensure_collection(client)
    return client


def _ensure_collection(client: MilvusClient) -> None:
    if not client.has_collection(COLLECTION):
        client.create_collection(COLLECTION, dimension=DIM, auto_id=True)
    client.load_collection(COLLECTION)


def _esc(s: str) -> str:
    """转义 Milvus 过滤表达式字符串中的特殊字符。"""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _npy_path(video_dir: Path) -> Path:
    return video_dir / "mert" / "mert_v1_95m_embeddings.npy"


def _video_vector(video_dir: Path) -> np.ndarray | None:
    """加载某视频的 MERT 帧向量，返回 L2 归一化的平均向量（视频级表示）。"""
    npy = _npy_path(video_dir)
    if not npy.is_file():
        return None
    emb = np.load(npy)
    vec = emb.mean(axis=0)
    return vec / (np.linalg.norm(vec) + 1e-12)


def _delete(client: MilvusClient, video_id: str) -> None:
    client.delete(COLLECTION, filter=f'video_id == "{_esc(video_id)}"')


def upsert_video(video_dir: Path | str) -> bool:
    """将单个视频结果目录的 MERT 向量写入库中（存在则先删后插）。

    video_dir 形如 results/master/<账号>/<视频名>/，返回是否成功写入。
    """
    video_dir = Path(video_dir)
    vec = _video_vector(video_dir)
    if vec is None:
        print(f"跳过（无 MERT 向量）: {video_dir.name}")
        return False
    npy = _npy_path(video_dir)
    client = get_client()
    _delete(client, video_dir.name)
    client.insert(COLLECTION, [{
        "vector": vec.astype(np.float32).tolist(),
        "account_id": video_dir.parent.name,
        "video_id": video_dir.name,
        "npy_mtime": npy.stat().st_mtime,
    }])
    print(f"已入库: {video_dir.parent.name}/{video_dir.name}")
    return True


def _existing_mtimes(client: MilvusClient) -> dict[str, float]:
    """读取库中已有 video_id -> npy_mtime 映射。"""
    rows = client.query(
        COLLECTION, filter="npy_mtime >= 0",
        output_fields=["video_id", "npy_mtime"], limit=10000,
    )
    return {r["video_id"]: r.get("npy_mtime", 0.0) for r in rows}


def ingest_new() -> tuple[int, int]:
    """增量导入：跳过未变化的视频，只入库新增/更新的。返回 (新增/更新数, 跳过数)。"""
    client = get_client()
    existing = _existing_mtimes(client)
    added = skipped = 0
    for account_dir in sorted(RESULTS_ROOT.iterdir()):
        if not account_dir.is_dir():
            continue
        for video_dir in sorted(account_dir.iterdir()):
            npy = _npy_path(video_dir)
            if not npy.is_file():
                continue
            if existing.get(video_dir.name) == npy.stat().st_mtime:
                skipped += 1
                continue
            if upsert_video(video_dir):
                added += 1
    print(f"增量导入完成: {added} 新增/更新, {skipped} 跳过")
    return added, skipped


def search_similar(account_id: str, video_id: str, top_k: int = 5):
    """以指定视频的平均向量检索最相似的其他视频。"""
    client = get_client()
    res = client.query(
        COLLECTION,
        filter=f'account_id == "{_esc(account_id)}" and video_id == "{_esc(video_id)}"',
        output_fields=["vector"],
        limit=1,
    )
    if not res:
        raise ValueError(f"未找到: {account_id}/{video_id}")
    hits = client.search(
        COLLECTION, [res[0]["vector"]], limit=top_k + 1,
        output_fields=["account_id", "video_id"],
    )[0]
    return [
        {"account_id": h["entity"]["account_id"],
         "video_id": h["entity"]["video_id"],
         "score": round(h["distance"], 4)}
        for h in hits if h["entity"]["video_id"] != video_id
    ][:top_k]


if __name__ == "__main__":
    ingest_new()
