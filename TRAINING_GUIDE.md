# Soujurou (静希草十郎) 日语音色训练指南

## 数据状态

| 项目 | 数值 |
|---|---|
| 原始素材 | 1846 个 mp3（122 分钟） |
| 切片 | 2216 段 @32kHz |
| 质检保留 | 1561 段（99.2 分钟） |
| ASR 标注保留 | **1522 段（97.2 分钟）** |
| 标注文件 | `Data/Soujurou/all_label.list`（`wav|spk|lang|text`） |

日语 g2p 走 pyopenjtalk（ja_userdic 134201 条）。音素表用 `symbols2`（732 符号），
句首/句末/重音边界标记 `^ $ #` 不在符号表内会被映射为 `UNK`（占 7.8%），
这是官方 pipeline 的固有行为，底模 s1v3 也是这样训的，不要改。

## 数据准备（阶段 0-3）

本文件的 `train_workflow.py` 从 `all_label.list` 开始，它由 `tools/` 下三个脚本产出：

| 脚本 | 做什么 | 产物 |
|---|---|---|
| 官方 `tools/slice_audio.py`（WebUI 0b 面板） | 按静音切段并重采样 32kHz | `slicer/audios/` 2216 段 |
| `tools/1-quality-gate.py` | 时长 / RMS / 静音比 / 削波闸门，淘汰文件移到 `slicer/rejected/` | `qc_report.csv`，保留 1561 段 |
| `tools/2-asr-label.py` | faster-whisper large-v3 日语转写 + 幻觉/置信度闸门 | `asr_metrics.csv`，保留 1522 段 |
| `tools/3-make-report.py` | 合并两份 CSV，写最终标注 | **`all_label.list`** + `dataset_report.csv/.txt` |

阈值、判定逻辑和踩过的坑（faster-whisper 的 CUDA 会因缺 `cublas64_12.dll` 回退 CPU）写在仓库 README 里。

## 工作流

`train_workflow.py` 把官方 6 步串起来，每步都能单独跑：

| 阶段 | 做什么 | 产物 |
|---|---|---|
| `text` | 日语音素提取 | `2-name2text.txt` |
| `hubert` | HuBERT 特征 + 32kHz wav | `4-cnhubert/` `5-wav32k/` |
| `sv` | 说话人嵌入（v2ProPlus 必需） | `7-sv_cn/` |
| `semantic` | 语义 token | `6-name2semantic.tsv` |
| `s1` | 训练 GPT | `logs_s1_v2ProPlus/` + `GPT_weights_v2ProPlus/` |
| `s2` | 训练 SoVITS | `logs_s2_v2ProPlus/` + `SoVITS_weights_v2ProPlus/` |

注意 `sv` 产物目录是 **`7-sv_cn`**（不是 6），`semantic` 是 `6-name2semantic.tsv`。

### 执行命令

```bash
cd D:\ai\GPT-SoVITS
.venv\Scripts\python.exe Data\Soujurou\train_workflow.py --stage all
```

常用参数：

```
--stage all|prep|train|text|hubert|sv|semantic|s1|s2
--parts 4          预处理并行进程数（默认 4，CPU 是瓶颈，8 物理核）
--s1-epochs 10     GPT 轮数
--s2-epochs 10     SoVITS 轮数
--s1-batch 8       GPT batch
--s2-batch 6       SoVITS batch（8 在 8GB 卡上峰值 7.8GB，太满）
--save-every-epoch 2
--force            重算已存在的产物
--dry-run          只打印命令
```

已完成的阶段会自动跳过（按产物数量判定，不是看目录在不在），
中断后重跑同一条命令即可续跑，不用从头来。

## 精简点（相比官方默认配置）

官方 `s1longer-v2.yaml` / `s2v2ProPlus.json` 是给几十万条大数据集写的，
直接拿来跑 97 分钟的小语料纯属浪费：

| 项 | 官方默认 | 本项目 | 原因 |
|---|---|---|---|
| s2 epoch | 100 | 10 | 97 分钟数据，官方 WebUI 在 8GB 卡上的默认也是 8 |
| s2 batch | 32 | 6 | 官方 WebUI 给 8GB 卡的默认就是 4；batch 8 实测峰值 7.8/8.2GB，留余量（但见「监控要点」：降到 6 其实没省出多少显存） |
| s2 步数 | 4800 | 2540 | 计算量降到约 1/2 |
| s1 epoch | 20 | 10 | |
| s1 batch | 8 | 8 | 8GB 卡够用，batch 8 兼顾显存和收敛 |
| ckpt 频率 | 每轮 | 每 2 轮 | 全量 ckpt 约 900MB/次，减少磁盘写入 |
| ckpt 保留 | 全部 | 只留最新 | `if_save_latest`，历史靠 half 权重快照挑 |
| 预处理 | 单进程 | 4 进程分片 | 瓶颈在 CPU 音频解码，不在 GPU |

学习率不用调：`WarmupCosineLRSchedule.step()` 里写死了 `lr = 0.002`，
yaml 里的 `lr/warmup_steps/decay_steps` 对 GPT 实际不起作用。

**实测耗时**（RTX 4060 Laptop 8GB，正式训练 10 epoch 的真实数字）：

| 阶段 | 实测 | 备注 |
|---|---|---|
| GPT (s1) 10 轮 | **13 分钟** | batch 8 |
| SoVITS (s2) 10 轮 | **87 分钟** | batch 6，约 8.7 分钟/轮 |
| 预处理 4 步 | 约 20 分钟 | 4 进程，实测 630 文件/分钟 |
| **合计** | **约 1 小时 40 分** | 官方默认配置要 5-9 小时 |

训练期间关掉浏览器和 WebUI 能明显省时间——实测 s2 后半程每步从 1.7s 掉到 3.0s，
就是显存被占满后开始抖动。

## Windows 注意事项

脚本已在子进程环境里注入 `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`。
不加的话 PyTorch Lightning 的 rich 进度条会在 GBK 控制台抛
`UnicodeEncodeError: 'gbk' codec can't encode character '\u2022'` 直接把训练打断。

`s2_train.py` 的参数是 `-c/--config`（JSON），`s1_train.py` 是 `-c/--config_file`（YAML），
两者不能混用——脚本已经分开传了。

## 训练后验证

```bash
cd D:\ai\GPT-SoVITS
.venv\Scripts\python.exe webui.py            # 主 WebUI（含 1C 推理页）
```

或者只开推理页：

```bash
.venv\Scripts\python.exe GPT_SoVITS/inference_webui_fast.py zh_CN
```

在推理页选择：

- 模型版本：v2ProPlus
- GPT 模型：`Soujurou-e10.ckpt`（在 `GPT_weights_v2ProPlus/`）
- SoVITS 模型：`Soujurou_e10_s*.pth`（在 `SoVITS_weights_v2ProPlus/`）
- 参考音频：从 `Data/Soujurou/slicer/audios/` 里挑一段干净的

先用训练集里的片段做参考（音色最稳），再用真实录音验证泛化。

脚本化验证走 `eval/`：`tts_test.py`（训练集参考合成 3 句）→ `verify.py`（RMS/峰值 + ASR 回读）
→ `generalization_test.py`（用 39 段从未训练的切片做参考）→ `generalization_readback.py` → `generalization_ab.py`（A/B 对比）。
注意 `TTS_Config` 要先构造再改字段（`update_version` / `t2s_weights_path` / `vits_weights_path`），
传扁平 dict 给构造函数会被忽略；`TTS.run` 是 generator，yield `(sr, int16 ndarray)`。

## 监控要点

- GPU 温度 < 85°C，笔记本散热有限；`nvidia-smi dmon -s pucm -d 10 -o TD` 看温度、显存和利用率
- **显存不是靠降 batch 省出来的**：`dmon` 实测 s2 batch 8 峰值 7841 MiB、batch 6 峰值 8163 MiB（总量 8188），
  占用由常驻模型和特征缓存主导，batch 只是小头。真要腾显存是关掉同时用 GPU 的程序（浏览器/WebUI），
  那才是每步从 1.7s 抖到 3.0s 的原因
- 训练全程 GPU 峰值温度 60°C，笔记本散热无压力
- 训练日志：`Data/Soujurou/logs_s1_v2ProPlus/`、`logs_s2_v2ProPlus/`
- 训练中断后重跑同一条命令会从最新 ckpt 续训

## 产物清单

```
Data/Soujurou/
├── all_label.list         # 最终标注（训练输入）
├── 2-name2text.txt        # 音素
├── 3-bert/                # 中文才有，日语音色为空目录
├── 4-cnhubert/            # HuBERT 特征
├── 5-wav32k/              # 32kHz wav 副本
├── 7-sv_cn/               # 说话人嵌入
├── 6-name2semantic.tsv    # 语义 token
├── runtime_s1.yaml        # 运行时生成的 GPT 配置
├── runtime_s2.json        # 运行时生成的 SoVITS 配置
├── logs_s1_v2ProPlus/     # GPT checkpoint
├── logs_s2_v2ProPlus/     # SoVITS checkpoint
├── slicer/audios/         # 切片后 wav
├── qc_report.csv          # 质检指标
├── asr_metrics.csv        # ASR 指标
└── dataset_report.txt     # 统计报告
```

## 故障排查

**显存不足**：把 `--s2-batch` 降到 4；s1 batch 降到 4。

**训练不收敛**：检查 `all_label.list` 里有没有 ASR 幻觉文本
（`asr_metrics.csv` 里 avg_logprob 明显偏低的行）；换一批参考音频。

**生成质量差**：把 `--s1-epochs` / `--s2-epochs` 提到 20-25 续训；
或换更干净的参考音频。

**`ModuleNotFoundError: resampy`**：`uv pip install resampy`（推理时需要）。

---

**最后更新**：2026-11-05
**状态**：训练完成（s1 + s2 各 10 epoch，12:25→14:06，实际 1 小时 40 分）；泛化测试与 ASR 回读质检通过；
权重 `Soujurou-e10.ckpt` / `Soujurou_e10_s2540.pth` 已发布在 https://github.com/si126048/soujurou-audio-test

本文件只覆盖阶段 4-6 的训练；阶段 0-3 的素材切片、音频质检、ASR 标注脚本在 `tools/`，
推理与泛化验证在 `eval/`，完整链路说明见仓库根 README。
