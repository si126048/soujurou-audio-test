import wave
from pathlib import Path

import numpy as np

EVAL = Path(r"D:\ai\GPT-SoVITS\Data\Soujurou\eval")
targets = {
    "test_0.wav": "こんにちは、今日もいい天気ですね。",
    "test_1.wav": "魔法使いの夜は、静かな夜に始まります。",
    "test_2.wav": "本当にありがとうございました。",
}

for name, tgt in targets.items():
    p = EVAL / name
    with wave.open(str(p), "rb") as wf:
        n, sr = wf.getnframes(), wf.getframerate()
        raw = wf.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    rms = float(np.sqrt((a**2).mean()))
    peak = float(np.abs(a).max())
    print(f"{name}  {n/sr:.2f}s @ {sr}Hz  rms={rms:.4f} peak={peak:.3f}  target={tgt}")

from faster_whisper import WhisperModel

model = WhisperModel(
    r"D:\ai\GPT-SoVITS\tools\asr\models\faster-whisper-large-v3",
    device="cpu",
    compute_type="int8",
)
print("\n--- ASR 回读 ---")
for name, tgt in targets.items():
    p = EVAL / name
    segs, info = model.transcribe(str(p), language="ja", beam_size=5)
    txt = "".join(s.text for s in segs).strip()
    print(f"{name}\n  目标: {tgt}\n  回读: {txt}\n  lang={info.language} prob={info.language_probability:.3f}\n")
