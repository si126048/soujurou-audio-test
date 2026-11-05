import os, csv, json
from collections import Counter

ROOT = r"D:\ai\GPT-SoVITS\Data\Soujurou"
QC = os.path.join(ROOT, "qc_report.csv")
ASR = os.path.join(ROOT, "asr_metrics.csv")
LIST = os.path.join(ROOT, "all_label.list")
MERGED = os.path.join(ROOT, "dataset_report.csv")
SUMMARY = os.path.join(ROOT, "dataset_report.txt")
AUD = os.path.join(ROOT, "slicer", "audios")

def readcsv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

qc = {r["file"]: r for r in readcsv(QC)}
asr = {r["file"]: r for r in readcsv(ASR)}

merged = []
for name, a in asr.items():
    q = qc.get(name, {})
    merged.append({
        "file": name,
        "dur_s": a["audio_dur_s"] or q.get("dur_s", ""),
        "rms_dbfs": q.get("rms_dbfs", ""),
        "silence_ratio": q.get("silence_ratio", ""),
        "avg_logprob": a["avg_logprob"],
        "no_speech_prob": a["no_speech_prob"],
        "chars_per_s": a["chars_per_s"],
        "qc": q.get("decision", "?"),
        "asr": a["decision"],
        "asr_reason": a["reason"],
        "text": a["text"],
    })

kept = [m for m in merged if m["asr"] == "keep"]
if merged:
    with open(MERGED, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)

with open(LIST, "w", encoding="utf-8") as f:
    for m in kept:
        f.write("%s|sojourou|JA|%s\n" % (os.path.join(AUD, m["file"]), m["text"]))

def stats(vals):
    vals = sorted(vals)
    if not vals:
        return {}
    n = len(vals)
    return {"n": n, "min": vals[0], "p10": vals[int(n*0.1)], "median": vals[n//2],
            "p90": vals[int(n*0.9)], "max": vals[-1],
            "mean": round(sum(vals)/n, 3), "total_min": round(sum(vals)/60, 2)}

durs = [float(m["dur_s"]) for m in kept if m["dur_s"]]
rms = [float(m["rms_dbfs"]) for m in kept if m["rms_dbfs"]]
lp = [float(m["avg_logprob"]) for m in kept if m["avg_logprob"]]
chars = sum(len(m["text"]) for m in kept)
text_all = " ".join(m["text"] for m in kept)

lines = []
lines.append("Soujurou (静希草十郎) 语料筛选与标注报告")
lines.append("=" * 62)
lines.append("原始素材      : 1846 个 mp3 = 122.1 分钟 (Data/Soujurou/raw_wavs, 微信 SoujurouVo.zip)")
lines.append("切片后        : %d 段 wav @32kHz (Data/Soujurou/slicer/audios)" % len(qc))
lines.append("质检淘汰      : %d" % sum(1 for r in qc.values() if r["decision"] == "drop"))
lines.append("ASR 淘汰      : %d" % sum(1 for m in merged if m["asr"] == "drop"))
lines.append("最终入库      : %d 条" % len(kept))
lines.append("")
lines.append("入库时长统计  : %s" % json.dumps(stats(durs), ensure_ascii=False))
lines.append("RMS(dBFS)     : %s" % json.dumps(stats(rms), ensure_ascii=False))
lines.append("ASR 置信度    : %s" % json.dumps(stats(lp), ensure_ascii=False))
lines.append("总字符数      : %d  (约 %d 字)" % (chars, chars))
lines.append("平均语速      : %.2f 字符/秒" % (chars / max(sum(durs), 1e-9)))
lines.append("")
lines.append("质检淘汰原因  : %s" % Counter(r["reason"] for r in qc.values() if r["decision"] == "drop").most_common())
lines.append("ASR 淘汰原因  : %s" % Counter(r["asr_reason"] for r in merged if r["asr"] == "drop").most_common())
lines.append("")
lines.append("样本 (前 25 条):")
for m in kept[:25]:
    lines.append("  [%-5.2fs] %s" % (float(m["dur_s"]), m["text"]))
lines.append("")
lines.append("产物:")
lines.append("  all_label.list     %d 行  (wav|spk|lang|text)" % len(kept))
lines.append("  dataset_report.csv 每条片段的质检+ASR 指标")
lines.append("  qc_report.csv      音频侧质检指标")
lines.append("  asr_metrics.csv    ASR 侧指标与文本")

with open(SUMMARY, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
