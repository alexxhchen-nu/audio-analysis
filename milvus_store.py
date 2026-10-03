"""Milvus Lite 向量存储：导入 results/master 下各视频的 MERT 向量，支持相似度检索。

用法:
    from milvus_store import ingest_all, search_similar
    ingest_all()                                # 全量导入（幂等，重复运行会重建集合）
    hits = search_similar("人民日报", "<视频名>")  # 以某视频为查询，找最近邻
"""
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "master"
DB_PATH = PROJECT_ROOT / "results" / "milvus_mert.db"
COLLECTION = "mert_embeddings"
DIM = 768  # MERT-v1-95M 输出维度


def get_client() -> MilvusClient:
    client = MilvusClient(str(DB_PATH))
    if client.has_collection(COLLECTION):
        client.load_collection(COLLECTION)
    return client


def _video_vector(video_dir: Path) -> np.ndarray | None:
    """加载某视频的 MERT 帧向量，返回 L2 归一化的平均向量（视频级表示）。"""
    npy = video_dir / "mert" / "mert_v1_95m_embeddings.npy"
    if not npy.is_file():
        return None
    emb = np.load(npy)
    vec = emb.mean(axis=0)
    return vec / (np.linalg.norm(vec) + 1e-12)


def ingest_all() -> int:
    """扫描 results/master/<账号>/<视频>/mert/ 并重建集合，返回写入条数。"""
    data = []
    for account_dir in sorted(RESULTS_ROOT.iterdir()):
        if not account_dir.is_dir():
            continue
        for video_dir in sorted(account_dir.iterdir()):
            vec = _video_vector(video_dir)
            if vec is None:
                continue
            data.append({
                "vector": vec.astype(np.float32).tolist(),
                "account_id": account_dir.name,
                "video_id": video_dir.name,
            })

    client = get_client()
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
    client.create_collection(COLLECTION, dimension=DIM, auto_id=True)
    if data:
        client.insert(COLLECTION, data)
    print(f"ingested {len(data)} videos -> {DB_PATH} :: {COLLECTION}")
    return len(data)


def search_similar(account_id: str, video_id: str, top_k: int = 5):
    """以指定视频的平均向量检索最相似的其他视频。"""
    client = get_client()
    res = client.query(
        COLLECTION,
        filter=f'account_id == "{account_id}" and video_id == "{video_id}"',
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
    ingest_all()
