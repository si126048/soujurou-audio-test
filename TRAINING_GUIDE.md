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
| s2 batch | 32 | 6 | 官方 WebUI 给 8GB 卡的默认就是 4；batch 8 实测峰值 7.8/8.2GB，留余量 |
| s2 步数 | 4800 | 2540 | 计算量降到约 1/2 |
| s1 epoch | 20 | 10 | |
| s1 batch | 8 | 8 | 8GB 卡够用，batch 8 兼顾显存和收敛 |
| ckpt 频率 | 每轮 | 每 2 轮 | 全量 ckpt 约 900MB/次，减少磁盘写入 |
| ckpt 保留 | 全部 | 只留最新 | `if_save_latest`，历史靠 half 权重快照挑 |
| 预处理 | 单进程 | 4 进程分片 | 瓶颈在 CPU 音频解码，不在 GPU |

学习率不用调：`WarmupCosineLRSchedule.step()` 里写死了 `lr = 0.002`，
yaml 里的 `lr/warmup_steps/decay_steps` 对 GPT 实际不起作用。

**实测耗时**（RTX 4060 Laptop 8GB，单轮冒烟跑出来的真实数字）：

| 阶段 | 实测 | 10 轮合计 |
|---|---|---|
| GPT (s1) | 约 4 分钟/轮 | 约 40 分钟 |
| SoVITS (s2) | 约 8 分钟/轮（191 步 @2.4s/it，含浏览器抢显存时的抖动） | 约 80 分钟 |

- 预处理 4 步：约 20 分钟（4 进程，实测 630 文件/分钟）
- **合计约 2.5 小时**（官方默认配置要 5-9 小时）

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

## 监控要点

- GPU 温度 < 85°C，笔记本散热有限；`nvidia-smi` 看温度和显存
- s2 batch 6 实测峰值约 6.5GB；batch 8 会顶到 7.8GB，余量太小
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
**状态**：数据就绪；s1 / s2 各跑通 1 轮冒烟（权重落盘正确、配置加载正确）；正式训练待启动
