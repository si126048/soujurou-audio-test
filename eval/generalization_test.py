import csv
import json
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
OUT = DATA / "eval" / "generalization"
OUT.mkdir(parents=True, exist_ok=True)

GPT_CKPT = ROOT / "GPT_weights_v2ProPlus" / "Soujurou-e10.ckpt"
SOVITS_CKPT = ROOT / "SoVITS_weights_v2ProPlus" / "Soujurou_e10_s2540.pth"

# ---------- 1. 选出从未进过训练集的切片 ----------
used = set()
for line in open(DATA / "all_label.list", encoding="utf8").read().strip("\n").split("\n"):
    used.add(Path(line.split("|")[0]).name)

report = {r["file"]: r for r in csv.DictReader(open(DATA / "dataset_report.csv", encoding="utf-8-sig"))}
qc = {r["file"]: r for r in csv.DictReader(open(DATA / "qc_report.csv", encoding="utf-8-sig"))}

cands = []
for p in sorted((DATA / "slicer" / "audios").glob("*.wav")):
    if p.name in used:
        continue
    r = report.get(p.name, {})
    q = qc.get(p.name, {})
    dur = float(r.get("dur_s") or 0)
    nsp = float(r.get("no_speech_prob") or 1)
    rms = float(q.get("rms_dbfs") or -99)
    lp = float(r.get("avg_logprob") or -9)
    cands.append(dict(path=p, name=p.name, dur=dur, nsp=nsp, rms=rms, lp=lp))

# 参考音频要：有实际语音(低 no_speech)、电平正常、时长 3-6s
cands.sort(key=lambda c: (c["nsp"], -c["rms"]))
refs = []
for c in cands:
    if c["nsp"] <= 0.70 and -26 <= c["rms"] <= -12 and 2.8 <= c["dur"] <= 6.5:
        refs.append(c)
    if len(refs) == 3:
        break
print("候选(从未训练):", len(cands))
for c in refs:
    print(f"  REF {c['name'][:50]:50s} dur={c['dur']:.2f}s rms={c['rms']:.1f}dB nsp={c['nsp']:.3f}")

# ---------- 2. 给参考音重新做 ASR 拿 prompt 文本 ----------
from faster_whisper import WhisperModel

asr = WhisperModel(
    str(ROOT / "tools" / "asr" / "models" / "faster-whisper-large-v3"),
    device="cpu",
    compute_type="int8",
)
for c in refs:
    segs, info = asr.transcribe(str(c["path"]), language="ja", beam_size=5)
    c["prompt_text"] = "".join(s.text for s in segs).strip()
    print(f"  REF 文本: {c['prompt_text']}")

# ---------- 3. 载入模型 ----------
cfg = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
cfg.device = torch.device("cuda:0")
cfg.is_half = True
cfg.update_version("v2ProPlus")
cfg.t2s_weights_path = str(GPT_CKPT)
cfg.vits_weights_path = str(SOVITS_CKPT)
tts = TTS(cfg)

TEXTS = [
    "君は何を恐れているんだ。",
    "月光が静かな水面を照らしている。",
    "ええ、責任を取らせてもらうよ。",
]


def synth(ref, text, tag, ref_text_free=False, seed=1234):
    inputs = {
        "text": text,
        "text_lang": "all_ja",
        "ref_audio_path": str(ref["path"]),
        "aux_ref_audio_paths": [],
        "prompt_text": "" if ref_text_free else ref["prompt_text"],
        "prompt_lang": "all_ja",
        "top_k": 15,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": 1.0,
        "ref_text_free": ref_text_free,
        "split_bucket": True,
        "fragment_interval": 0.3,
        "seed": seed,
        "keep_random": False,
        "use_cuda_graph": False,
        "use_flash_attention": False,
        "repetition_penalty": 1.35,
    }
    for sr, audio in tts.run(inputs):
        p = OUT / f"{tag}.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.tobytes())
        return p, sr, len(audio) / sr


manifest = []
for i, ref in enumerate(refs):
    for j, text in enumerate(TEXTS):
        p, sr, dur = synth(ref, text, f"ref{i}_text{j}")
        manifest.append(dict(file=p.name, ref=ref["name"], ref_dur=ref["dur"],
                             text=text, dur=round(dur, 2)))
        print("wrote", p.name, f"{dur:.2f}s")

# 无 prompt 文本模式（纯靠说话人嵌入）
for i, ref in enumerate(refs):
    p, sr, dur = synth(ref, TEXTS[1], f"ref{i}_free", ref_text_free=True)
    manifest.append(dict(file=p.name, ref=ref["name"], ref_dur=ref["dur"],
                         text=TEXTS[1] + " [ref_text_free]", dur=round(dur, 2)))
    print("wrote", p.name, f"{dur:.2f}s (free)")

json.dump(manifest, open(OUT / "manifest.json", "w", encoding="utf8"), ensure_ascii=False, indent=2)
print("VRAM peak: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
print("DONE")
