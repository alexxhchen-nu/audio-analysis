#!/usr/bin/env python3
"""从 R2 下载账号视频，超长文件名自动截断（ext4 单文件名 255 字节限制）。

截断策略：保留开头的标题前缀 + 末尾的 _<视频ID>.mp4（若存在），
中间以 ... 连接，保证总字节数 <= 200。
已存在的文件跳过（按文件名判断幂等）。
"""
import json
import subprocess
import sys
from pathlib import Path

REMOTE = "cloudfare:pika-audio"
ACCOUNTS = {
    "news/banyuetan": "半月谈",
    "news/guangmingribao": "光明日报",
    "news/jingjiribao": "经济日报",
    "news/yangshixinwen": "央视新闻",
    "military/人民陆军--主页": "人民陆军",
    "military/南海舰队--主页": "南海舰队",
}
VIDEO_ROOT = Path("/home/ubuntu/music/video")
MAX_BYTES = 200


def truncate_name(name: str) -> str:
    if len(name.encode()) <= MAX_BYTES:
        return name
    stem, suffix = name, ""
    # 保留末尾 _<id>.mp4 或 <id>.mp4
    for sep in ("_", " "):
        head, _, tail = name.rpartition(sep)
        if tail.endswith(".mp4") and tail[:-4].isdigit():
            stem, suffix = head, sep + tail
            break
    else:
        if name.endswith(".mp4"):
            stem, suffix = name[:-4], ".mp4"
    suffix_b = len(suffix.encode())
    budget = MAX_BYTES - suffix_b - 3  # 3 = "..."
    out = []
    used = 0
    for ch in stem:
        b = len(ch.encode())
        if used + b > budget:
            break
        out.append(ch)
        used += b
    return "".join(out) + "..." + suffix


def main() -> None:
    only = set(sys.argv[1:])
    for remote_path, account in ACCOUNTS.items():
        if only and account not in only:
            continue
        dest_dir = VIDEO_ROOT / account
        dest_dir.mkdir(parents=True, exist_ok=True)
        listing = json.loads(subprocess.run(
            ["rclone", "lsjson", f"{REMOTE}/{remote_path}", "--files-only"],
            capture_output=True, check=True, text=True).stdout)
        todo = []
        for item in listing:
            if not item["Name"].lower().endswith(".mp4"):
                continue
            local = truncate_name(item["Name"])
            if (dest_dir / local).is_file():
                continue
            todo.append((item["Name"], local))
        print(f"{account}: {len(todo)} 待下载 / 共 {len(listing)}", flush=True)
        for i, (remote_name, local) in enumerate(todo, 1):
            subprocess.run(
                ["rclone", "copyto",
                 f"{REMOTE}/{remote_path}/{remote_name}",
                 str(dest_dir / local)],
                check=True)
            if i % 20 == 0:
                print(f"  {account}: {i}/{len(todo)}", flush=True)
    print("ALL_DOWNLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()
