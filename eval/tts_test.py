import csv
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\ai\GPT-SoVITS")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

DATA = ROOT / "Data" / "Soujurou"
OUT = DATA / "eval"
OUT.mkdir(exist_ok=True)

GPT_CKPT = ROOT / "GPT_weights_v2ProPlus" / "Soujurou-e10.ckpt"
SOVITS_CKPT = ROOT / "SoVITS_weights_v2ProPlus" / "Soujurou_e10_s2540.pth"

labels = {}
with open(DATA / "all_label.list", encoding="utf8") as f:
    for line in f.read().strip("\n").split("\n"):
        wav, spk, lang, text = line.split("|")
        labels[Path(wav).name] = text

metrics = []
with open(DATA / "asr_metrics.csv", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        if row.get("decision") != "keep":
            continue
        dur = float(row.get("audio_dur_s") or 0)
        lp = float(row.get("avg_logprob") or -9)
        nsp = float(row.get("no_speech_prob") or 1)
        if 4.0 <= dur <= 8.0 and lp > -0.35 and nsp < 0.4 and row["file"] in labels:
            metrics.append((lp, dur, row["file"]))
metrics.sort()
print("candidate refs:", len(metrics))
ref_file = metrics[0][2]
ref_wav = DATA / "slicer" / "audios" / ref_file
ref_text = labels[ref_file]
print("ref:", ref_file, "| dur", metrics[0][1], "| logprob", metrics[0][0])
print("ref text:", ref_text)

cfg = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
cfg.device = torch.device("cuda:0")
cfg.is_half = True
cfg.update_version("v2ProPlus")
cfg.t2s_weights_path = str(GPT_CKPT)
cfg.vits_weights_path = str(SOVITS_CKPT)
tts = TTS(cfg)
print("version:", tts.configs.version, "| half:", tts.configs.is_half, "| device:", tts.configs.device)

TEXTS = [
    "こんにちは、今日もいい天気ですね。",
    "魔法使いの夜は、静かな夜に始まります。",
    "本当にありがとうございました。",
]

for i, text in enumerate(TEXTS):
    inputs = {
        "text": text,
        "text_lang": "all_ja",
        "ref_audio_path": str(ref_wav),
        "aux_ref_audio_paths": [],
        "prompt_text": ref_text,
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
        p = OUT / f"test_{i}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        print("wrote", p, f"{len(audio)/sr:.2f}s @ {sr}Hz")

print("VRAM peak: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
print("DONE")
