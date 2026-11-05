import json
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(r"D:\ai\GPT-SoVITS")
OUT = ROOT / "Data" / "Soujurou" / "eval" / "generalization"
manifest = json.load(open(OUT / "manifest.json", encoding="utf8"))

model = WhisperModel(
    str(ROOT / "tools" / "asr" / "models" / "faster-whisper-large-v3"),
    device="cpu",
    compute_type="int8",
)

for m in manifest:
    p = OUT / m["file"]
    with wave.open(str(p), "rb") as wf:
        n, sr = wf.getnframes(), wf.getframerate()
        a = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768
    rms = float(np.sqrt((a**2).mean()))
    peak = float(np.abs(a).max())
    segs, info = model.transcribe(str(p), language="ja", beam_size=5)
    txt = "".join(s.text for s in segs).strip()
    print(f"{m['file']:18s} {n/sr:5.2f}s rms={rms:.3f} pk={peak:.2f} lang={info.language}/{info.language_probability:.2f}")
    print(f"   目标: {m['text']}")
    print(f"   回读: {txt}")
