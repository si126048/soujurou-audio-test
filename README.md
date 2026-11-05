# 静希草十郎 · GPT-SoVITS v2ProPlus 日语全链路工作流

从 1846 个原始 mp3 到可推理音色模型的完整流程与脚本。GPT-SoVITS v2ProPlus，日语，10 epoch 微调。

语料：《魔法使之夜》静希草十郎，1522 段 / 97.2 分钟。
硬件：RTX 4060 Laptop 8GB，i7 8 物理核，Windows 11，Python 3.10 + uv venv。
实测耗时：特征预处理 ~20 min + s1 13 min + s2 87 min，**训练部分合计 1 小时 40 分**（官方默认配置的 s2 要 5-9 小时）。数据准备各步未逐项计时，ASR 标注是其中最有 CPU 压力的一步。

## 全流程一览

| 阶段 | 脚本 | 输入 | 产物 |
|---|---|---|---|
| 0 切片 | 官方 `tools/slice_audio.py`（WebUI 0b 面板） | 1846 mp3 / 122.1 min | 2216 段 wav @32kHz |
| 1 音频质检 | `tools/1-quality-gate.py` | `slicer/audios/` | `qc_report.csv` + `slicer/rejected/`，保留 1561 段 / 99.2 min |
| 2 ASR 标注 | `tools/2-asr-label.py` | 通过质检的 wav | `asr_metrics.csv` + `slicer/label/audios.list`，保留 1522 段 |
| 3 合并报告 | `tools/3-make-report.py` | 上两步 CSV | **`all_label.list`**（训练输入）+ `dataset_report.csv/.txt` |
| 4 特征预处理 | `train_workflow.py --stage prep` | `all_label.list` + `slicer/audios` | `2-name2text.txt` / `4-cnhubert` / `5-wav32k` / `7-sv_cn` / `6-name2semantic.tsv` |
| 5 训 GPT | `train_workflow.py --stage s1` | 阶段 4 产物 | `GPT_weights_v2ProPlus/Soujurou-e10.ckpt`（13 min） |
| 6 训 SoVITS | `train_workflow.py --stage s2` | 阶段 4 产物 | `SoVITS_weights_v2ProPlus/Soujurou_e10_s2540.pth`（87 min） |
| 7 推理验证 | `eval/*.py` | 权重 + 参考音频 | `test_*.wav` / `generalization/*.wav` + ASR 回读质检 |

阶段 3 之后就可以直接用官方 WebUI 走完剩余流程；这里的脚本只是把 6 步串起来并砍掉无用配置。

## 0. 准备

 clone 到 GPT-SoVITS 项目里，本仓库脚本对应放在 `Data/Soujurou/` 下：

```
GPT-SoVITS/
├── Data/Soujurou/
│   ├── raw_wavs/              # 1846 个原始 mp3
│   ├── slicer/audios/         # 切片输出（阶段 0）
│   ├── slicer/rejected/       # 质检淘汰（阶段 1 移入）
│   ├── slicer/label/audios.list
│   ├── tools/                 # 阶段 1-3 脚本
│   ├── eval/                  # 阶段 7 脚本
│   ├── train_workflow.py      # 阶段 4-6
│   ├── all_label.list         # 最终标注（不公开，见下）
│   └── qc_report.csv / asr_metrics.csv / dataset_report.*
├── GPT_weights_v2ProPlus/     # 训练产物
└── SoVITS_weights_v2ProPlus/
```

脚本里的路径是**硬编码绝对路径**（每个文件顶部的 `ROOT` / `MODEL`），换机器或换实验只需改这几行：

| 文件 | 需要改的量 |
|---|---|
| `tools/1-quality-gate.py` | `ROOT` |
| `tools/2-asr-label.py` | `ROOT`、`MODEL`（faster-whisper 模型目录） |
| `tools/3-make-report.py` | `ROOT`；`SUMMARY` 里的素材统计文案是写死的 |
| `train_workflow.py` | 顶部常量块：`EXP_NAME` / `VERSION` / `LANGUAGE` / `PYTHON` / 预训练模型路径 |
| `eval/*.py` | `ROOT`、`GPT_CKPT` / `SOVITS_CKPT` |

需要日语底模 `s1v3.ckpt`、`pretrained_models/chinese-hubert-base`、`chinese-roberta-wwm-ext-large`、`pretrained_models/v2Pro/s2Gv2ProPlus.pth` + `s2Dv2ProPlus.pth`、`pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`，以及 `tools/asr/models/faster-whisper-large-v3`。

## 1. 切片

用官方切分工具即可（WebUI 的 0b 语音切分面板，或直接命令行）：

```bash
python -s tools/slice_audio.py Data/Soujurou/raw_wavs Data/Soujurou/slicer \
    -34 4000 300 10 500 0.9 0.25 0 1
#                       └ threshold └ min_length └ min_interval └ hop_size └ max_sil_kept └ _max └ alpha └ i_part └ all_part
```

参数是 WebUI 面板的默认值，多进程时按 `i_part`/`all_part` 分片。输出结构是 `<输出根目录>/audios/` + `label/`，所以上面的命令会写到 `Data/Soujurou/slicer/audios/`。

1846 个 mp3（122.1 分钟）切成 2216 段。

## 2. 音频质检闸门

```bash
python Data/Soujurou/tools/1-quality-gate.py
```

逐段算时长、RMS(dBFS)、峰值、削波比例、静音帧比例、首尾静音长度，按阈值淘汰并把文件**移到** `slicer/rejected/`（不是删除，可回滚）。判定标准：

| 指标 | 阈值 | 淘汰原因 |
|---|---|---|
| 时长 | 2.0 - 12.0 s | `too_short` / `too_long` |
| RMS | ≥ -45 dBFS | `too_quiet` |
| 静音比例 | ≤ 50%（30 ms 帧 / 10 ms hop，<-40 dBFS 记为静音） | `too_silent` |
| 削波采样占比 | ≤ 0.5%（\|x\| ≥ 0.99） | `clipping` |

2216 → **1561 段保留 / 99.2 分钟**。结果写 `qc_report.csv`。

## 3. ASR 标注

```bash
python Data/Soujurou/tools/2-asr-label.py
```

faster-whisper large-v3，强制 `language="ja"`，`beam_size=5`，开 VAD（`min_silence_duration_ms=700`），`condition_on_previous_text=False`。每段拿三个可信度信号做闸门：

| 指标 | 淘汰阈值 | 原因 |
|---|---|---|
| `avg_logprob` | < -1.0 | `low_conf` |
| `no_speech_prob` | > 0.6 | `no_speech` |
| 字符/秒 | < 1.0 或 > 14.0 | `too_few_chars` / `too_many_chars` |
| 命中幻觉黑名单 | - | `hallucination`（「ご視聴ありがとうございました」「.amara.org」「未经许可不得转载」等） |
| 空文本 | - | `empty` |

1561 → **1522 段保留 / 97.2 分钟**，同时产出 `asr_metrics.csv`（每段的置信度与文本）和 `slicer/label/audios.list`。

**这一步是本机 CPU 跑的**：CTranslate2 的 CUDA 需要 `cublas64_12.dll`，不在 PATH 就报 `RuntimeError: Library cublas64_12.dll is not found`——而 `torch.cuda.is_available()` 仍返回 True。脚本已改成"真的构造一次再回退 `device="cpu", compute_type="int8"`"，别只信 `is_available()`。

## 4. 合并成训练标注

```bash
python Data/Soujurou/tools/3-make-report.py
```

把质检和 ASR 两份 CSV 按文件名合并成 `dataset_report.csv`，只保留 ASR 判定 `keep` 的行写成 `all_label.list`（格式 `wav绝对路径|说话人|语种|文本`），并输出 `dataset_report.txt` 统计（时长/RMS/置信度分布、淘汰原因计数、平均语速、前 25 条样本）。

`all_label.list` 是整个流程唯一的训练输入，也是唯一含有原作台词的文件，所以**没有放进仓库**。自己跑一遍阶段 1-3 就会生成。

## 5-6. 预处理 + 训练

```bash
cd D:\ai\GPT-SoVITS
.venv\Scripts\python.exe Data\Soujurou\train_workflow.py --stage all
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--stage` | all | `all/prep/train/text/hubert/sv/semantic/s1/s2` |
| `--parts` | 4 | 预处理并行进程数（瓶颈在 CPU 音频解码，不在 GPU） |
| `--s1-epochs` / `--s2-epochs` | 10 / 10 | 轮数 |
| `--s1-batch` / `--s2-batch` | 8 / 6 | batch |
| `--save-every-epoch` | 2 | 快照间隔 |
| `--force` / `--dry-run` | - | 强制重算 / 只打印命令 |

已完成的阶段按**产物数量**判定并跳过，中断后重跑同一条命令即可续跑。四个预处理子阶段：

- `text` 音素提取 → `2-name2text.txt`（日语走 pyopenjtalk，`^ $ #` 边界标记不在符号表内会映射成 UNK，约 7.8%，是官方 pipeline 固有行为，别改）
- `hubert` → `4-cnhubert/*.pt` + `5-wav32k/*.wav`
- `sv` → `7-sv_cn/*.pt`（v2ProPlus 必需，目录是 **7** 不是 6）
- `semantic` → `6-name2semantic.tsv`（带 `item_name\tsemantic_audio` 表头）

分片产物会像 WebUI 一样合并成单文件。详细参数取舍、实测每轮耗时、排错见 `TRAINING_GUIDE.md`。

## 7. 推理与泛化验证

| 脚本 | 做什么 |
|---|---|
| `eval/tts_test.py` | 从 `asr_metrics.csv` 挑一段高置信训练集切片当参考，合成 3 句 → `test_*.wav` |
| `eval/verify.py` | RMS / 峰值 + faster-whisper 回读比对目标文本 |
| `eval/generalization_test.py` | 用**从未进训练集**的 3 段切片做参考 × 3 句新文本 + 3 段 `ref_text_free`（不喂 prompt 文本） |
| `eval/generalization_readback.py` | 按 `manifest.json` 逐条 ASR 回读 |
| `eval/generalization_ab.py` | A/B：训练集内参考 vs 未见参考，同 4 句文本 |

泛化测试能成立的关键：`slicer/audios`（1561 段）比 `all_label.list`（1522 段）多出的 **39 段确实没参与训练**，正好当 held-out 参考。

推理加载权重的写法（`TTS_Config` 必须先构造再改字段，传扁平 dict 给构造函数会被忽略）：

```python
cfg = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
cfg.device = torch.device("cuda:0")
cfg.is_half = True
cfg.update_version("v2ProPlus")
cfg.t2s_weights_path = str(GPT_CKPT)
cfg.vits_weights_path = str(SOVITS_CKPT)
tts = TTS(cfg)
```

`TTS.run(inputs)` 是 **generator**，yield `(sr, np.int16 ndarray)`；`text_lang` / `prompt_lang` 用原始码 `all_ja`；`text_split_method` 用字符串 `cut0`-`cut5`。

## 泛化测试音频

| 文件 | 说明 |
|---|---|
| `AB_train_ref{0-3}.wav` | 用**训练集内**参考音合成 |
| `AB_unseen_ref{0-3}.wav` | 用**从未见过**的参考音合成（泛化测试） |
| `ref{0-2}_text{0-2}.wav` | 3 个未见参考音 × 3 句新文本 |
| `ref{0-2}_free.wav` | 不喂 prompt 文本，纯靠说话人嵌入 |

测试文本：

- 句 0: 君は何を恐れているんだ。
- 句 1: 月光が静かな水面を照らしている。
- 句 2: ええ、責任を取らせてもらうよ。
- 句 3 (长句): さあ、扉の向こうには、两人待ち合わせの夜が待っている。

质检：ASR 回读全部正确（仅标点/语气词微小差异），语种判定 ja 概率 1.00，rms 0.17-0.25，peak 0.92-0.97。已知一处瑕疵：`ref0_free.wav` 把 月光が 回读成 月光間。

## 训练好的模型

`Soujurou-e10.ckpt`（GPT，149 MB）和 `Soujurou_e10_s2540.pth`（SoVITS，165 MB）走 Git LFS 放在仓库根，用法见 `MODEL.md`。`runtime_s1.yaml` / `runtime_s2.json` 是这次训练实际用的配置快照，可以直接拿去复现同样的参数。

## 踩过的坑

- **`s1_train.py` 参数是 `-c/--config_file`（YAML），`s2_train.py` 是 `-c/--config`（JSON）**，混用直接崩。
- **Windows GBK 控制台**会让 PyTorch Lightning 的 rich 进度条抛 `UnicodeEncodeError: 'gbk' codec can't encode character '\u2022'` 打断训练。子进程环境必须注入 `PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1`。
- **降 batch 不是 8GB 卡的显存杠杆**。s2 batch 8 → 6 实测峰值 7841 vs 8163 MiB（总 8188），几乎没变——占用由常驻模型和特征缓存主导。真正影响速度的是有没有别的程序抢 GPU：浏览器/WebUI 在跑时每步从 1.7s 抖到 3.0s。
- **GPT 学习率调不动**：`AR/modules/lr_schedulers.py` 的 `WarmupCosineLRSchedule.step()` 写死 `lr = 0.002`，YAML 里的 `lr`/`warmup_steps`/`decay_steps` 对 GPT 无效。
- **官方 epoch/batch 是给几十万条语料写的**，97 分钟数据用 s2 100 epoch × batch 32 纯属浪费；本项目 10 × 6，总步数 4800 → 2540。
- **`save_weight_dir` 是 `runtime_s2.json` 的顶层键**，不在 `train` 里。
- **faster-whisper 的 CUDA 依赖 `cublas64_12.dll`**，本机没在 PATH，见阶段 3。
