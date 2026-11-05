# 静希草十郎 v2ProPlus 模型与训练工作流

GPT-SoVITS v2ProPlus，日语，10 epoch 微调
语料：1522 段 / 97.2 分钟（《魔法使之夜》静希草十郎）

## 仓库内容

| 文件 | 说明 |
|---|---|
| `MODEL.md` | 模型使用说明（WebUI / 脚本推理） |
| `TRAINING_GUIDE.md` | 完整训练指南（数据、参数、排错） |
| `train_workflow.py` | 一键训练工作流脚本 |
| `Soujurou-e10.ckpt` | GPT 权重 |
| `Soujurou_e10_s2540.pth` | SoVITS 权重 |
| `runtime_s1.yaml` / `runtime_s2.json` | 运行时生成的训练配置 |
| `*.wav` | 泛化测试音频 |

## 训练工作流

```bash
cd GPT-SoVITS
.venv\Scripts\python.exe Data/Soujurou/train_workflow.py --stage all
```

6 个阶段：`text` → `hubert` → `sv` → `semantic` → `s1` → `s2`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--stage` | all | `all/prep/train/text/hubert/sv/semantic/s1/s2` |
| `--parts` | 4 | 预处理并行进程数（瓶颈在 CPU 音频解码） |
| `--s1-epochs` | 10 | GPT 轮数 |
| `--s2-epochs` | 10 | SoVITS 轮数 |
| `--s1-batch` | 8 | GPT batch |
| `--s2-batch` | 6 | SoVITS batch（8GB 卡） |
| `--save-every-epoch` | 2 | 快照间隔 |
| `--force` | - | 强制重算已存在产物 |
| `--dry-run` | - | 只打印命令 |

支持断点续跑：中断后重跑同一条命令即可从最新产物继续。

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

质检：ASR 回读全部正确（仅标点/语气词微小差异），语种判定 ja 概率 1.00。
