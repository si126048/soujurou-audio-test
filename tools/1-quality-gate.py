import os, sys, csv, math, shutil, glob
import numpy as np
import soundfile as sf

ROOT = r"D:\ai\GPT-SoVITS\Data\Soujurou"
AUD = os.path.join(ROOT, "slicer", "audios")
REJ = os.path.join(ROOT, "slicer", "rejected")
REPORT = os.path.join(ROOT, "qc_report.csv")

MIN_DUR, MAX_DUR = 2.0, 12.0
MIN_RMS_DB = -45.0
MAX_SILENCE_RATIO = 0.50
MAX_CLIP_RATIO = 0.005
SIL_DB = -40.0
FRAME = 0.03   # 30 ms analysis frame
HOP = 0.01     # 10 ms hop

os.makedirs(REJ, exist_ok=True)

def frame_stats(x, sr):
    n = int(FRAME * sr)
    hop = int(HOP * sr)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    nf = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(nf)[:, None]
    fr = x[idx]
    e = np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(e)
    return db, hop

rows = []
files = sorted(glob.glob(os.path.join(AUD, "*.wav")))
print("sliced files:", len(files), flush=True)

for i, p in enumerate(files):
    name = os.path.basename(p)
    try:
        x, sr = sf.read(p, dtype="float32", always_2d=False)
    except Exception as e:
        rows.append([name, 0, -99, 0, 0, 1, 0, 0, "drop", "read_error:%r" % (e,)])
        continue
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    peak = float(np.abs(x).max()) if len(x) else 0.0
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean())) if len(x) else 0.0
    rms_db = 20 * math.log10(rms + 1e-12)
    clip_ratio = float((np.abs(x) >= 0.99).mean()) if len(x) else 1.0

    db, hop = frame_stats(x, sr)
    sil = db < SIL_DB
    silence_ratio = float(sil.mean()) if len(sil) else 1.0
    lead = 0
    for v in sil:
        if v:
            lead += hop
        else:
            break
    trail = 0
    for v in sil[::-1]:
        if v:
            trail += hop
        else:
            break

    reasons = []
    if dur < MIN_DUR:
        reasons.append("too_short")
    if dur > MAX_DUR:
        reasons.append("too_long")
    if rms_db < MIN_RMS_DB:
        reasons.append("too_quiet")
    if silence_ratio > MAX_SILENCE_RATIO:
        reasons.append("too_silent")
    if clip_ratio > MAX_CLIP_RATIO:
        reasons.append("clipping")
    decision = "drop" if reasons else "keep"
    if reasons:
        shutil.move(p, os.path.join(REJ, name))
    rows.append([name, round(dur, 3), round(rms_db, 2), round(peak, 4),
                 round(clip_ratio, 5), round(silence_ratio, 4),
                 round(lead / sr, 3), round(trail / sr, 3), decision, ";".join(reasons)])
    if (i + 1) % 200 == 0:
        print("  qc %d/%d" % (i + 1, len(files)), flush=True)

with open(REPORT, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "dur_s", "rms_dbfs", "peak", "clip_ratio",
                "silence_ratio", "lead_sil_s", "trail_sil_s", "decision", "reason"])
    w.writerows(rows)

keep = sum(1 for r in rows if r[8] == "keep")
drop = len(rows) - keep
tot = sum(r[1] for r in rows if r[8] == "keep")
print("QC done: keep=%d drop=%d  kept_duration=%.1f min" % (keep, drop, tot / 60), flush=True)
from collections import Counter
print("drop reasons:", Counter(r[9] for r in rows if r[8] == "drop").most_common(), flush=True)
