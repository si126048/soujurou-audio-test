# 静希草十郎 v2ProPlus 模型

## 模型信息

- **角色**: 静希草十郎（《魔法使之夜》）
- **语言**: 日语 (ja)
- **版本**: v2ProPlus
- **训练轮数**: 10 epochs
- **训练语料**: 1522 段 / 97.2 分钟
- **训练耗时**: 1 小时 40 分 (RTX 4060 Laptop 8GB)

## 文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `Soujurou-e10.ckpt` | 149MB | GPT 权重 (Text2Semantic) |
| `Soujurou_e10_s2540.pth` | 165MB | SoVITS 权重 (VITS vocoder) |
| `runtime_s1.yaml` | - | GPT 训练配置 |
| `runtime_s2.json` | - | SoVITS 训练配置 |

## 使用方法

### 1. WebUI 推理

```bash
cd GPT-SoVITS
.venv\Scripts\python.exe GPT_SoVITS/inference_webui_fast.py zh_CN
```

在推理页选择：
- **模型版本**: v2ProPlus
- **GPT 模型**: `Soujurou-e10.ckpt`
- **SoVITS 模型**: `Soujurou_e10_s2540.pth`
- **参考音频**: 从训练集 `Data/Soujurou/slicer/audios/` 选一段清晰的

### 2. 脚本推理

```python
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
import torch

cfg = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
cfg.device = torch.device("cuda:0")
cfg.is_half = True
cfg.update_version("v2ProPlus")
cfg.t2s_weights_path = "Soujurou-e10.ckpt"
cfg.vits_weights_path = "Soujurou_e10_s2540.pth"

tts = TTS(cfg)

inputs = {
    "text": "こんにちは、今日もいい天気ですね。",
    "text_lang": "all_ja",
    "ref_audio_path": "reference.wav",
    "prompt_text": "参考音频的文本",
    "prompt_lang": "all_ja",
    "top_k": 15,
    "top_p": 1,
    "temperature": 1,
    "text_split_method": "cut5",
    "batch_size": 1,
    "speed_factor": 1.0,
    "ref_text_free": False,
    "split_bucket": True,
    "fragment_interval": 0.3,
    "seed": 1234,
    "keep_random": False,
    "use_cuda_graph": False,
    "use_flash_attention": False,
    "repetition_penalty": 1.35,
}

for sr, audio in tts.run(inputs):
    import wave
    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
```

## 训练配置摘要

- **s1 (GPT)**: 10 epochs, batch 8, 每 2 轮保存快照
- **s2 (SoVITS)**: 10 epochs, batch 6, 每 2 轮保存快照
- **学习率**: GPT 固定 0.002 (代码写死，yaml 无效)
- **半精度**: fp16 (is_half=True)

## 泛化测试

测试音频在 `eval/` 目录，包含：
- A/B 对照（训练集参考音 vs 未见参考音）
- 无 prompt 文本模式（纯靠说话人嵌入）
- ASR 回读验证（全部通过）

## 硬件要求

- **GPU**: RTX 4060 Laptop 8GB 或更高
- **显存**: 推理约 1.7GB，训练约 7-8GB
- **Python**: 3.10+
- **依赖**: 见 GPT-SoVITS 仓库 requirements

## 许可证

模型基于 GPT-SoVITS v2ProPlus 微调，遵循原项目许可证。
训练数据来自《魔法使之夜》游戏音频，仅供学习研究。
