<div align="center">

# 🎧✨ Little Audio Garden

### A curious, careful toolkit for exploring music and audio in short videos

🌷 *Listen closely · compare fairly · keep the human in the loop* 🌷

[🇬🇧 English](#english) · [🇨🇳 简体中文](#简体中文)

</div>

<a id="english"></a>

## 🇬🇧 English

Welcome to **Little Audio Garden**! 🌱 This research project explores how audio in short videos is put together: music, voices, acoustic features, and similarities between clips. It helps us ask better questions; it does **not** automatically decide what a clip means or what someone intended.

### 🌼 What you can do

- 🎬 Analyze audio from a **local video or audio file**—the reusable workflow does not download media.
- 🎛️ Create estimated `vocals` and `no_vocals` stems with Demucs while keeping the original audio.
- 📊 Measure interpretable sound features with Librosa.
- 🧠 Create timestamped music embeddings with MERT and explore nearest neighbors or clusters.
- 🎹 Optionally estimate note events with Basic Pitch.
- 💗 Try a supervised music-emotion baseline using DEAM's human valence/arousal ratings and compare Librosa features with MERT embeddings.
- 🔎 Inspect results, listen to candidate matches, and keep every analysis traceable to its source and timestamps.

### 🧪 Emotion pilot: an honest first test

The reproducible 50-song pilot is in [`scripts/deam_emotion_pilot.py`](scripts/deam_emotion_pilot.py). It uses 5-second, non-overlapping windows beginning at 15 seconds, aligns each window to DEAM's 500-ms valence/arousal ratings, computes 62 Librosa features and 768-dimensional MERT embeddings, and holds out entire songs for testing.

The initial fixed-seed split contains 40 training songs and 10 test songs (300 windows total). On this small holdout, both feature sets performed better than a training-mean baseline, but these exploratory scores are **not** a validated emotion model and do not establish transfer to short-video audio. See the [pilot report](results/emotion/deam_pilot/README.md) and raw [metrics](results/emotion/deam_pilot/metrics.csv). A different random seed or more songs may change the results.

Run it from the repository root:

```bash
python scripts/deam_emotion_pilot.py --n-tracks 50 --seed 42
```

The script expects DEAM assets under the Git-ignored `data/emotion_reference/` folder and a locally cached MERT checkpoint. To explicitly allow Transformers to fetch MERT model code and weights:

```bash
python scripts/deam_emotion_pilot.py --n-tracks 50 --seed 42 --download-model
```

`--download-model` uses `trust_remote_code=True`: review the upstream [MERT model card](https://huggingface.co/m-a-p/MERT-v1-95M) and model code before permitting that download/execution. The pilot writes its manifest, features, predictions, metrics, and configuration under `results/emotion/deam_pilot/`; the large embedding matrix (`.npy`) is ignored by Git.

### 🧰 Tech stack

| Tool | What it does here |
|---|---|
| 🐍 **Python + Jupyter** | Reusable workflows, experiments, and research notes |
| 🎼 **Librosa** | MFCC, chroma, RMS/energy, spectral features, and music analysis |
| 🧠 **PyTorch + Transformers + MERT** | Learned, timestamped music embeddings |
| 🥁 **Demucs** | Estimated vocal/accompaniment source separation |
| 🧪 **scikit-learn** | Scaling, grouped holdout evaluation, and ridge-regression baselines |
| 🎹 **Basic Pitch** *(optional)* | Candidate note-event and MIDI transcription |
| 🎞️ **FFmpeg** | Local video-audio extraction and media conversion |

Most project analysis is Python/Jupyter. MERT and Basic Pitch are optional model stages and may need additional packages, model downloads, and a suitable GPU/CPU.

### 🚀 Quick start

1. Open [`notebooks/master.ipynb`](notebooks/master.ipynb) and set `INPUT_MEDIA` to a file you placed in `video/`.
2. Run the notebook cells in order. Video audio is extracted locally with FFmpeg; there is no automatic downloader.
3. Find outputs under `results/master/<input-name>/`.
4. Use [`notebooks/test.ipynb`](notebooks/test.ipynb) for scratch experiments and [`notebooks/notes.ipynb`](notebooks/notes.ipynb) for methodology and the work journal.

The master notebook documents its baseline package setup. FFmpeg must also be available on the machine. PyTorch installation depends on the available CPU/GPU; use the [official installation selector](https://pytorch.org/get-started/locally/) for your environment.

### 🗺️ Repository map

| Path | What's inside |
|---|---|
| [`notebooks/master.ipynb`](notebooks/master.ipynb) | Reusable local-media analysis workflow |
| [`notebooks/test.ipynb`](notebooks/test.ipynb) | Clean scratch notebook |
| [`notebooks/notes.ipynb`](notebooks/notes.ipynb) | Methods, interpretation caveats, and work journal |
| [`scripts/deam_emotion_pilot.py`](scripts/deam_emotion_pilot.py) | Reproducible DEAM feature/label alignment and baseline evaluation |
| [`results/emotion/deam_pilot/`](results/emotion/deam_pilot/) | Pilot report, metrics, predictions, and selected songs |
| [`results/corpus_clustering/no_vocals_mert/`](results/corpus_clustering/no_vocals_mert/) | Exploratory cross-video MERT clustering outputs |
| [`data/emotion_reference/`](data/emotion_reference/) | Local reference datasets; intentionally not tracked by Git |
| [`video/`](video/) | Local source-media drop folder; not tracked by Git |

### 🧭 Responsible interpretation

- An embedding or cluster means **similar under that representation**, not the same meaning, emotion, or intent.
- DEAM ratings describe human judgments in that dataset. A DEAM-trained model still needs human-reviewed testing on this project's own clips.
- `no_vocals` is Demucs's estimated accompaniment stem, not guaranteed speech-free audio or ground truth.
- Emotion ratings and model outputs do not reveal a listener's or creator's private internal state.
- Clustering can group by source video, recording, encoding, loudness, or production rather than shared musical content.
- Preserve originals, cite dataset sources, check their licenses, and report uncertainty.

### 💾 Media and privacy

Put local source media in `video/`; generated analysis files go under `results/`. For durable work, use a persistent volume and keep a backup. The local `video/` folder and the current container filesystem are not a backup.

Research datasets, their terms, and large audio are deliberately kept out of Git. Download or redistribute them only in accordance with each dataset's license and terms. The DEAM and MTG-Jamendo reference data currently used by this project are documented in [`notebooks/notes.ipynb`](notebooks/notes.ipynb).

---

<a id="简体中文"></a>

## 🇨🇳 简体中文

欢迎来到 **Little Audio Garden（小小音频花园）**！🌱 这是一个探索短视频音频构成的研究项目：音乐、人声、可解释声学特征，以及不同片段之间的相似性。它帮助我们提出和检验问题，**不会**自动判定音频的含义或创作者意图。

### 🌼 可以做什么

- 🎬 分析**本地视频或音频文件**；可复用流程不会自动下载媒体。
- 🎛️ 使用 Demucs 估计 `vocals` 和 `no_vocals` 音轨，同时保留原音频。
- 📊 使用 Librosa 计算可解释的声音特征。
- 🧠 使用 MERT 生成带时间戳的音乐向量，并探索近邻和聚类。
- 🎹 可选使用 Basic Pitch 估计音符事件。
- 💗 用 DEAM 人工效价／唤醒度评分做监督式音乐情绪基线，比较 Librosa 特征与 MERT 向量。
- 🔎 检查结果、试听候选匹配，并保留来源与时间戳，确保分析可追溯。

### 🧪 情绪试验：先做一个诚实的小测试

可复现的 50 首歌曲试验位于 [`scripts/deam_emotion_pilot.py`](scripts/deam_emotion_pilot.py)。它从第 15 秒开始切分 5 秒、不重叠的窗口，将每个窗口与 DEAM 每 500 毫秒的效价／唤醒度评分对齐，计算 62 维 Librosa 特征和 768 维 MERT 向量，并按整首歌留出测试集。

固定种子的初步划分为 40 首训练、10 首测试，共 300 个窗口。在这次小型留出测试中，两类特征的表现都优于训练集均值基线；但这只是探索性结果，**不能视为已验证的情绪模型，也不能证明模型能迁移到短视频音频**。详见[试验报告](results/emotion/deam_pilot/README.md)和[原始指标](results/emotion/deam_pilot/metrics.csv)。更换随机种子或增加歌曲后，结果可能变化。

在仓库根目录运行：

```bash
python scripts/deam_emotion_pilot.py --n-tracks 50 --seed 42
```

脚本需要 `data/emotion_reference/` 中的本地 DEAM 文件，以及已缓存的 MERT 模型。若希望 Transformers 下载 MERT 模型代码和权重，请明确运行：

```bash
python scripts/deam_emotion_pilot.py --n-tracks 50 --seed 42 --download-model
```

`--download-model` 会启用 `trust_remote_code=True`；允许下载和执行前，请检查 [MERT 模型卡](https://huggingface.co/m-a-p/MERT-v1-95M)及其上游模型代码。脚本会把曲目清单、特征、预测、指标和配置写入 `results/emotion/deam_pilot/`；较大的 embedding 矩阵（`.npy`）已被 Git 忽略。

### 🧰 技术栈

| 工具 | 在本项目中的用途 |
|---|---|
| 🐍 **Python + Jupyter** | 可复用工作流、实验与研究笔记 |
| 🎼 **Librosa** | MFCC、chroma、RMS／能量、频谱特征和音乐分析 |
| 🧠 **PyTorch + Transformers + MERT** | 学习型、带时间戳的音乐向量 |
| 🥁 **Demucs** | 估计人声／伴奏分离 |
| 🧪 **scikit-learn** | 特征标准化、按歌曲留出评估和岭回归基线 |
| 🎹 **Basic Pitch**（可选） | 候选音符事件和 MIDI 转录 |
| 🎞️ **FFmpeg** | 本地提取视频音轨和媒体格式转换 |

项目分析主要基于 Python/Jupyter。MERT 和 Basic Pitch 属于可选模型步骤，可能需要额外依赖、模型下载及合适的 GPU／CPU。

### 🚀 快速开始

1. 打开 [`notebooks/master.ipynb`](notebooks/master.ipynb)，将 `INPUT_MEDIA` 设置为放在 `video/` 中的文件名。
2. 按顺序运行 notebook 单元。视频音轨由 FFmpeg 在本地提取；流程不会自动下载媒体。
3. 在 `results/master/<输入名称>/` 查看分析输出。
4. 使用 [`notebooks/test.ipynb`](notebooks/test.ipynb) 做临时实验，在 [`notebooks/notes.ipynb`](notebooks/notes.ipynb) 查看方法说明、解释注意事项和工作日志。

主 notebook 中有基础依赖安装说明；运行机器还需要 FFmpeg。PyTorch 安装方式取决于 CPU/GPU 环境，请参考[官方安装选择器](https://pytorch.org/get-started/locally/)。

### 🗺️ 仓库导航

| 路径 | 内容 |
|---|---|
| [`notebooks/master.ipynb`](notebooks/master.ipynb) | 可复用的本地媒体分析流程 |
| [`notebooks/test.ipynb`](notebooks/test.ipynb) | 空白实验 notebook |
| [`notebooks/notes.ipynb`](notebooks/notes.ipynb) | 研究方法、结果解释注意事项与工作日志 |
| [`scripts/deam_emotion_pilot.py`](scripts/deam_emotion_pilot.py) | DEAM 特征／标签对齐与基线评估脚本 |
| [`results/emotion/deam_pilot/`](results/emotion/deam_pilot/) | 试验报告、指标、预测和曲目清单 |
| [`results/corpus_clustering/no_vocals_mert/`](results/corpus_clustering/no_vocals_mert/) | 跨视频 MERT 聚类探索结果 |
| [`data/emotion_reference/`](data/emotion_reference/) | 本地参考数据集；不会由 Git 跟踪 |
| [`video/`](video/) | 本地源媒体目录；不会由 Git 跟踪 |

### 🧭 谨慎解释

- 向量或聚类表示“在该表示空间中相似”，不等于含义、情绪或意图相同。
- DEAM 评分反映该数据集中的人类判断。使用 DEAM 训练后，仍需用项目自己的音频和人工评分检验适用性。
- `no_vocals` 是 Demucs 估计的伴奏音轨，不保证完全无人声，也不是 ground truth。
- 情绪评分和模型输出不能揭示听众或创作者的私人内心状态。
- 聚类可能按视频来源、录音、编码、响度或制作差异分组，而非按共同音乐内容分组。
- 保留原件、注明数据来源、核对许可，并报告不确定性。

### 💾 媒体与隐私

将本地源媒体放入 `video/`，分析结果放在 `results/`。重要数据应放在持久化存储中并另做备份；本地 `video/` 目录和当前容器文件系统都不能替代备份。

研究数据集、许可条款和大型音频文件有意排除在 Git 之外。下载或再分发前请遵守相应数据集的许可和使用条款。项目当前使用的 DEAM 与 MTG-Jamendo 参考资料记录在 [`notebooks/notes.ipynb`](notebooks/notes.ipynb)。

---

🌱 **Thanks for listening with care.** · **感谢你认真聆听。** 🎧
