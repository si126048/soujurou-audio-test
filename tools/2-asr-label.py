import os, sys, csv, math, time, json
import numpy as np

ROOT = r"D:\ai\GPT-SoVITS\Data\Soujurou"
AUD = os.path.join(ROOT, "slicer", "audios")
LABEL_DIR = os.path.join(ROOT, "slicer", "label")
METRICS = os.path.join(ROOT, "asr_metrics.csv")
MODEL = r"D:\ai\GPT-SoVITS\tools\asr\models\faster-whisper-large-v3"

os.makedirs(LABEL_DIR, exist_ok=True)

from faster_whisper import WhisperModel

# faster-whisper 的 CTranslate2 自带 CUDA 依赖，缺 cublas64_12.dll 时 torch 报告 cuda
# 可用但一调用就抛 RuntimeError，所以真的构造一次再回退 CPU，不要只信 is_available()。
try:
    model = WhisperModel(MODEL, device="cuda", compute_type="float16")
    device = "cuda"
except Exception as e:
    print("cuda unavailable for ctranslate2 (%r), using cpu int8" % (e,), flush=True)
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    device = "cpu"
print("device:", device, flush=True)

HALLUC = [
    "ご視聴ありがとうございました", "ご視聴ありがとうございます", "字幕視聴者", ".amara.org",
    "thanks for watching", "thank you for watching", "字幕をMER", "おわり", "終わり",
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目", "明鏡與點點欄目", "www.youtube.com",
    "未经许可不得转载", "大家好", "欢迎收看", "bv号",
]

files = sorted(f for f in os.listdir(AUD) if f.lower().endswith(".wav"))
print("files to ASR:", len(files), flush=True)

rows = []
t0 = time.time()
done = 0
for name in files:
    p = os.path.join(AUD, name)
    try:
        segments, info = model.transcribe(
            audio=p,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700),
            language="ja",
            condition_on_previous_text=False,
        )
        segs = list(segments)
        text = "".join(s.text for s in segs).strip()
        dur = sum(s.end - s.start for s in segs)
        if segs:
            w = [max(s.end - s.start, 1e-6) for s in segs]
            avg_lp = sum(s.avg_logprob * wi for s, wi in zip(segs, w)) / sum(w)
            nsp = max(s.no_speech_prob for s in segs)
        else:
            avg_lp, nsp = -99.0, 1.0
        audio_dur = info.duration if hasattr(info, "duration") else 0.0
        low = text.lower()
        hall = any(h.lower() in low for h in HALLUC)
        nchar = len([c for c in text if not c.isspace()])
        cps = nchar / dur if dur > 0 else 0.0

        reasons = []
        if not text:
            reasons.append("empty")
        if hall:
            reasons.append("hallucination")
        if avg_lp < -1.0:
            reasons.append("low_conf")
        if nsp > 0.6:
            reasons.append("no_speech")
        if text and cps < 1.0:
            reasons.append("too_few_chars")
        if text and cps > 14.0:
            reasons.append("too_many_chars")
        decision = "drop" if reasons else "keep"
        rows.append([name, round(audio_dur, 3), round(dur, 3), round(avg_lp, 3),
                     round(nsp, 3), round(cps, 2), info.language, decision,
                     ";".join(reasons), text])
    except Exception as e:
        rows.append([name, 0, 0, -99, 1, 0, "?", "drop", "error:%r" % (e,), ""])
    done += 1
    if done % 100 == 0:
        el = time.time() - t0
        print("  asr %d/%d  %.1f fps  eta %.1f min" % (done, len(files), done / el,
              (len(files) - done) / (done / el) / 60), flush=True)

with open(METRICS, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "audio_dur_s", "speech_dur_s", "avg_logprob", "no_speech_prob",
                "chars_per_s", "lang", "decision", "reason", "text"])
    w.writerows(rows)

kept = [r for r in rows if r[7] == "keep"]
lst = os.path.join(LABEL_DIR, "audios.list")
with open(lst, "w", encoding="utf-8") as f:
    for r in kept:
        f.write("%s|%s|%s|%s\n" % (os.path.join(AUD, r[0]), "sojourou", "JA", r[9]))

from collections import Counter
print("ASR done in %.1f min" % ((time.time() - t0) / 60), flush=True)
print("keep=%d drop=%d" % (len(kept), len(rows) - len(kept)), flush=True)
print("drop reasons:", Counter(r[8] for r in rows if r[7] == "drop").most_common(), flush=True)
print("list:", lst, flush=True)
