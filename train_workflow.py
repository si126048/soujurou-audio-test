# -*- coding: utf-8 -*-
"""
Soujurou (静希草十郎) v2ProPlus 微调工作流（精简版）

用法（必须用项目 venv 解释器运行，脚本内部按需切子进程）:
    D:\\ai\\GPT-SoVITS\\.venv\\Scripts\\python.exe Data\\Soujurou\\train_workflow.py --stage all

    --stage all     预处理 + 训练（默认）
    --stage prep    只做 4 步预处理
    --stage train   只做 s1/s2 训练
    --stage text|hubert|sv|semantic|s1|s2   单步
    --dry-run       只打印将要执行的命令

阶段产物:
    text     -> Data/Soujurou/2-name2text.txt
    hubert   -> Data/Soujurou/4-cnhubert/*.pt + 5-wav32k/*.wav
    sv       -> Data/Soujurou/7-sv_cn/*.pt
    semantic -> Data/Soujurou/6-name2semantic.tsv
    s1       -> Data/Soujurou/logs_s1_v2ProPlus/ + GPT_weights_v2ProPlus/Soujourou-e*.ckpt
    s2       -> Data/Soujurou/logs_s2_v2ProPlus/ + SoVITS_weights_v2ProPlus/Soujourou_e*.pth
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
EXP_NAME = "Soujurou"
VERSION = "v2ProPlus"
LANGUAGE = "ja"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

DATA_DIR = ROOT / "Data" / EXP_NAME
INP_TEXT = DATA_DIR / "all_label.list"
INP_WAV_DIR = DATA_DIR / "slicer" / "audios"

PRETRAINED_MODELS = ROOT / "GPT_SoVITS" / "pretrained_models"
BERT_DIR = PRETRAINED_MODELS / "chinese-roberta-wwm-ext-large"
CNHUBERT_DIR = PRETRAINED_MODELS / "chinese-hubert-base"
SV_CKPT = PRETRAINED_MODELS / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt"
PRETRAINED_S1 = PRETRAINED_MODELS / "s1v3.ckpt"
PRETRAINED_S2G = PRETRAINED_MODELS / "v2Pro" / "s2Gv2ProPlus.pth"
PRETRAINED_S2D = PRETRAINED_MODELS / "v2Pro" / "s2Dv2ProPlus.pth"

S1_CFG_SRC = ROOT / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
S2_CFG_SRC = ROOT / "GPT_SoVITS" / "configs" / f"s2{VERSION}.json"
S1_CFG_OUT = DATA_DIR / "runtime_s1.yaml"
S2_CFG_OUT = DATA_DIR / "runtime_s2.json"

GPT_WEIGHT_DIR = ROOT / f"GPT_weights_{VERSION}"
SOVITS_WEIGHT_DIR = ROOT / f"SoVITS_weights_{VERSION}"
S1_LOG_DIR = DATA_DIR / f"logs_s1_{VERSION}"
S2_LOG_DIR = DATA_DIR / f"logs_s2_{VERSION}"

STAGE_ORDER = ["text", "hubert", "sv", "semantic", "s1", "s2"]
ARGS = None


def base_env(**over):
    env = {
        **os.environ,
        "exp_name": EXP_NAME,
        "version": VERSION,
        "opt_dir": str(DATA_DIR),
        "inp_text": str(INP_TEXT),
        "inp_wav_dir": str(INP_WAV_DIR),
        "is_half": "True",
        "bert_pretrained_dir": str(BERT_DIR),
        "cnhubert_base_dir": str(CNHUBERT_DIR),
        "sv_path": str(SV_CKPT),
        "pretrained_s2G": str(PRETRAINED_S2G),
        "s2config_path": str(S2_CFG_SRC),
        "_CUDA_VISIBLE_DEVICES": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "hz": "25hz",
        # Windows 下 PL 的 rich 进度条在 GBK 控制台会抛 UnicodeEncodeError
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "GPT_SoVITS")]),
    }
    env.update({k: str(v) for k, v in over.items()})
    return env


def run(cmd, desc, env, check=True):
    print(f"\n{'=' * 70}\n[{time.strftime('%H:%M:%S')}] {desc}\n{'=' * 70}")
    print("$ " + " ".join(str(c) for c in cmd))
    if ARGS.dry_run:
        return 0
    result = subprocess.run([str(c) for c in cmd], env=env, cwd=str(ROOT))
    if check and result.returncode != 0:
        print(f"\n[FAIL] {desc} (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n[OK] {desc}")
    return result.returncode


def run_parallel(commands, desc, envs):
    print(f"\n{'=' * 70}\n[{time.strftime('%H:%M:%S')}] {desc}\n{'=' * 70}")
    for cmd in commands:
        print("$ " + " ".join(str(c) for c in cmd))
    if ARGS.dry_run:
        return 0
    procs = [subprocess.Popen([str(c) for c in cmd], env=env, cwd=str(ROOT)) for cmd, env in zip(commands, envs)]
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"\n[FAIL] {desc} ({codes})")
        sys.exit(1)
    print(f"\n[OK] {desc}")
    return 0


def merge_parts(pattern, out_path, header=None):
    """把 i_part 分片合并成单个文件（与 WebUI 1A/1C 步骤一致）"""
    parts = sorted(DATA_DIR.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"没有找到分片: {pattern}")
    lines = []
    for part in parts:
        with open(part, "r", encoding="utf8") as f:
            lines += f.read().strip("\n").split("\n")
        part.unlink()
    with open(out_path, "w", encoding="utf8") as f:
        if header:
            f.write(header + "\n")
        f.write("\n".join(lines) + "\n")
    print(f"[merge] {len(lines)} 行 -> {out_path}")


def stage_text():
    out = DATA_DIR / "2-name2text.txt"
    if out.exists() and not ARGS.force:
        print(f"[skip] {out.name} 已存在")
        return
    run(
        [PYTHON, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"],
        "步骤 1/4: 音素提取 -> 2-name2text.txt",
        base_env(i_part=0, all_parts=1),
    )
    merge_parts("2-name2text-*.txt", out)


def label_count():
    with open(INP_TEXT, "r", encoding="utf8") as f:
        return len(f.read().strip("\n").split("\n"))


def done_count(subdir, suffix):
    d = DATA_DIR / subdir
    return len([n for n in os.listdir(d) if n.endswith(suffix)]) if d.exists() else 0


def stage_hubert():
    expected = label_count()
    n = done_count("4-cnhubert", ".pt")
    if n >= expected and not ARGS.force:
        print(f"[skip] 4-cnhubert/ 已完成 ({n}/{expected})")
        return
    print(f"[resume] 4-cnhubert/ 已有 {n}/{expected}")
    parts = max(1, ARGS.parts)
    cmds, envs = [], []
    for i in range(parts):
        cmds.append([PYTHON, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"])
        envs.append(base_env(i_part=i, all_parts=parts))
    run_parallel(cmds, f"步骤 2/4: HuBERT 特征 + 32kHz wav（{parts} 进程）", envs)


def stage_sv():
    expected = label_count()
    n = done_count("7-sv_cn", ".pt")
    if n >= expected and not ARGS.force:
        print(f"[skip] 7-sv_cn/ 已完成 ({n}/{expected})")
        return
    print(f"[resume] 7-sv_cn/ 已有 {n}/{expected}")
    parts = max(1, ARGS.parts)
    cmds, envs = [], []
    for i in range(parts):
        cmds.append([PYTHON, "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"])
        envs.append(base_env(i_part=i, all_parts=parts))
    run_parallel(cmds, f"步骤 3/4: 说话人嵌入 -> 7-sv_cn/（{parts} 进程）", envs)


def stage_semantic():
    out = DATA_DIR / "6-name2semantic.tsv"
    if out.exists() and not ARGS.force:
        print(f"[skip] {out.name} 已存在")
        return
    parts = max(1, ARGS.parts)
    cmds, envs = [], []
    for i in range(parts):
        cmds.append([PYTHON, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"])
        envs.append(base_env(i_part=i, all_parts=parts))
    run_parallel(cmds, f"步骤 4/4: 语义 token（{parts} 进程）", envs)
    merge_parts("6-name2semantic-*.tsv", out, header="item_name\tsemantic_audio")


def build_s1_config():
    with open(S1_CFG_SRC, "r", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = ARGS.s1_epochs
    cfg["train"]["batch_size"] = ARGS.s1_batch
    cfg["train"]["save_every_n_epoch"] = ARGS.save_every_epoch
    cfg["train"]["if_save_latest"] = True
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["if_dpo"] = False
    cfg["train"]["exp_name"] = EXP_NAME
    cfg["train"]["half_weights_save_dir"] = str(GPT_WEIGHT_DIR)
    cfg["data"]["num_workers"] = ARGS.workers
    cfg["pretrained_s1"] = str(PRETRAINED_S1)
    cfg["train_semantic_path"] = str(DATA_DIR / "6-name2semantic.tsv")
    cfg["train_phoneme_path"] = str(DATA_DIR / "2-name2text.txt")
    cfg["output_dir"] = str(S1_LOG_DIR)
    with open(S1_CFG_OUT, "w", encoding="utf8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return cfg


def build_s2_config():
    with open(S2_CFG_SRC, "r", encoding="utf8") as f:
        cfg = json.load(f)
    cfg["train"]["epochs"] = ARGS.s2_epochs
    cfg["train"]["batch_size"] = ARGS.s2_batch
    cfg["train"]["pretrained_s2G"] = str(PRETRAINED_S2G)
    cfg["train"]["pretrained_s2D"] = str(PRETRAINED_S2D)
    cfg["train"]["if_save_latest"] = True
    cfg["train"]["if_save_every_weights"] = True
    cfg["train"]["save_every_epoch"] = ARGS.save_every_epoch
    cfg["train"]["gpu_numbers"] = "0"
    cfg["train"]["log_interval"] = 50
    cfg["model"]["version"] = VERSION
    cfg["data"]["exp_dir"] = str(DATA_DIR)
    cfg["s2_ckpt_dir"] = str(DATA_DIR)
    cfg["save_weight_dir"] = str(SOVITS_WEIGHT_DIR)
    cfg["name"] = EXP_NAME
    cfg["version"] = VERSION
    with open(S2_CFG_OUT, "w", encoding="utf8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg


def stage_s1():
    GPT_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    S1_LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_s1_config()
    print(f"[s1] epochs={cfg['train']['epochs']} batch={cfg['train']['batch_size']} -> {S1_LOG_DIR}")
    run(
        [PYTHON, "-s", "GPT_SoVITS/s1_train.py", "--config_file", S1_CFG_OUT],
        "步骤 5/6: 训练 GPT (s1)",
        base_env(),
    )


def stage_s2():
    SOVITS_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    S2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_s2_config()
    print(f"[s2] epochs={cfg['train']['epochs']} batch={cfg['train']['batch_size']} -> {S2_LOG_DIR}")
    run(
        [PYTHON, "-s", "GPT_SoVITS/s2_train.py", "--config", S2_CFG_OUT],
        "步骤 6/6: 训练 SoVITS (s2)",
        base_env(),
    )


STAGES = {
    "text": stage_text,
    "hubert": stage_hubert,
    "sv": stage_sv,
    "semantic": stage_semantic,
    "s1": stage_s1,
    "s2": stage_s2,
}


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", help="all/prep/train 或单个阶段名")
    parser.add_argument("--force", action="store_true", help="忽略已存在产物，强制重算")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parts", type=int, default=4, help="预处理并行进程数（CPU 瓶颈，8 物理核建议 4）")
    parser.add_argument("--workers", type=int, default=4, help="s1 dataloader 线程数")
    parser.add_argument("--s1-epochs", type=int, default=10)
    parser.add_argument("--s1-batch", type=int, default=8)
    parser.add_argument("--s2-epochs", type=int, default=10)
    # 8GB 卡上 batch 8 实测峰值 7.8GB/8.2GB，余量不足，降到 6 留出安全边界
    parser.add_argument("--s2-batch", type=int, default=6)
    parser.add_argument("--save-every-epoch", type=int, default=2)
    ARGS = parser.parse_args()

    if not INP_TEXT.exists():
        print(f"[FAIL] 缺少标注文件 {INP_TEXT}")
        sys.exit(1)
    if not INP_WAV_DIR.exists():
        print(f"[FAIL] 缺少切片目录 {INP_WAV_DIR}")
        sys.exit(1)

    if ARGS.stage == "all":
        stages = STAGE_ORDER
    elif ARGS.stage == "prep":
        stages = STAGE_ORDER[:4]
    elif ARGS.stage == "train":
        stages = STAGE_ORDER[4:]
    elif ARGS.stage in STAGES:
        stages = [ARGS.stage]
    else:
        print(f"[FAIL] 未知阶段 {ARGS.stage}")
        sys.exit(1)

    print(f"实验: {EXP_NAME} | 版本: {VERSION} | 语言: {LANGUAGE}")
    print(f"阶段: {' -> '.join(stages)}")
    t0 = time.time()
    for name in stages:
        STAGES[name]()
    print(f"\n{'=' * 70}")
    print(f"全部完成，用时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"GPT 权重: {GPT_WEIGHT_DIR}")
    print(f"SoVITS 权重: {SOVITS_WEIGHT_DIR}")
    print(f"下一步: 启动推理页 http://127.0.0.1:9872 加载 {EXP_NAME} 模型试听")
    print("=" * 70)


if __name__ == "__main__":
    main()
