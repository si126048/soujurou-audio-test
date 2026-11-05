import json
import sys
import wave
from pathlib import Path

import torch

ROOT = Path(r"D:\ai\GPT-SoVITS")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

DATA = ROOT / "Data" / "Soujurou"
OUT = DATA / "eval" / "generalization"

cfg = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
cfg.device = torch.device("cuda:0")
cfg.is_half = True
cfg.update_version("v2ProPlus")
cfg.t2s_weights_path = str(ROOT / "GPT_weights_v2ProPlus" / "Soujurou-e10.ckpt")
cfg.vits_weights_path = str(ROOT / "SoVITS_weights_v2ProPlus" / "Soujurou_e10_s2540.pth")
tts = TTS(cfg)

TEXTS = [
    "君は何を恐れているんだ。",
    "月光が静かな水面を照らしている。",
    "ええ、責任を取らせてもらうよ。",
    "さあ、扉の向こうには，两人待ち合わせの夜が待っている。",
]

# A 组：训练集内的参考音（之前质检用的那段）
labels = {}
for line in open(DATA / "all_label.list", encoding="utf8").read().strip("\n").split("\n"):
    wav, spk, lang, text = line.split("|")
    labels[Path(wav).name] = text
import csv

cand = []
for row in csv.DictReader(open(DATA / "asr_metrics.csv", encoding="utf-8-sig")):
    if row.get("decision") != "keep":
        continue
    dur = float(row["audio_dur_s"] or 0)
    lp = float(row["avg_logprob"] or -9)
    nsp = float(row["no_speech_prob"] or 1)
    if 4.0 <= dur <= 8.0 and lp > -0.35 and nsp < 0.4 and row["file"] in labels:
        cand.append((lp, row["file"]))
cand.sort()
train_ref = DATA / "slicer" / "audios" / cand[0][1]
train_prompt = labels[cand[0][1]]
print("A组参考音(训练集内):", train_ref.name, "|", train_prompt)

# B 组：从未进过训练集的参考音
unseen_ref = DATA / "slicer" / "audios" / "3668_a30_2_5_0000.mp3_0000000000_0000101440.wav"
unseen_prompt = "店長、外回りの掃除終わりました。"
print("B组参考音(从未训练):", unseen_ref.name, "|", unseen_prompt)


def synth(ref, prompt, text, tag):
    inputs = {
        "text": text, "text_lang": "all_ja", "ref_audio_path": str(ref),
        "aux_ref_audio_paths": [], "prompt_text": prompt, "prompt_lang": "all_ja",
        "top_k": 15, "top_p": 1, "temperature": 1, "text_split_method": "cut5",
        "batch_size": 1, "speed_factor": 1.0, "ref_text_free": False,
        "split_bucket": True, "fragment_interval": 0.3, "seed": 1234,
        "keep_random": False, "use_cuda_graph": False, "use_flash_attention": False,
        "repetition_penalty": 1.35,
    }
    for sr, audio in tts.run(inputs):
        p = OUT / f"{tag}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        print("wrote", p.name, f"{len(audio)/sr:.2f}s")


for j, t in enumerate(TEXTS):
    synth(train_ref, train_prompt, t, f"AB_train_ref{j}")
    synth(unseen_ref, unseen_prompt, t, f"AB_unseen_ref{j}")
print("DONE")
