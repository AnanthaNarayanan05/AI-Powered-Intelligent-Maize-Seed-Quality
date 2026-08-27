"""Re-runs the Dataset B variety experiments on the group-aware split.

Why this exists (docs/09 section 6.9): the original Dataset B split placed
augmented copies of all 127 source seeds into all three splits, so every test
image had a same-seed sibling in training. This re-runs the identical four
ablations against manifest_dataset_b_grouped.csv, where each source seed lives in
exactly one split, and against a contrastive encoder pretrained on the train
split alone. Original artefacts are left untouched: everything written here
carries a _grouped suffix.
"""
import os, subprocess, sys, time

PY_EXE = os.path.abspath(".venv/Scripts/python.exe")
MANIFEST = "data_processed/manifest_dataset_b_grouped.csv"
ENCODER = "outputs/checkpoints/contrastive_encoder_b_grouped.pt"
LOG = "outputs/logs/grouped_b_runner.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def vram_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def settle(limit=1500, timeout=180):
    """A run launched the instant the previous one exits can fail to allocate:
    the dying process has not yet released its VRAM. Wait for the card to drain."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = vram_mb()
        if m < limit:
            return
        time.sleep(5)
    log(f"WARNING: VRAM still at {vram_mb()} MiB after {timeout}s, proceeding anyway")


def run(name, cmd, marker):
    done = f"outputs/checkpoints/.{marker}.done"
    if os.path.exists(done):
        log(f"SKIP {name} (already complete)")
        return True
    settle()
    log(f"START {name}")
    t0 = time.time()
    r = subprocess.run([PY_EXE, "-m"] + cmd)
    dt = (time.time() - t0) / 60
    if r.returncode != 0:
        log(f"FAIL  {name} exit={r.returncode} after {dt:.1f} min -- retrying once")
        settle()
        r = subprocess.run([PY_EXE, "-m"] + cmd)
        if r.returncode != 0:
            log(f"FAIL  {name} again exit={r.returncode}. Stopping.")
            return False
    open(done, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"DONE  {name} in {dt:.1f} min")
    return True


STAGES = [
    ("contrastive pretrain (train split only)", "b_grouped_pretrain",
     ["src.training.pretrain_contrastive", "--dataset", "b",
      "--manifest", MANIFEST, "--tag", "grouped"]),
    ("baseline", "b_grouped_baseline",
     ["src.training.train_variety", "--dataset", "b", "--experiment", "baseline",
      "--manifest", MANIFEST, "--tag", "baseline_grouped"]),
    ("attention_only", "b_grouped_attention",
     ["src.training.train_variety", "--dataset", "b", "--experiment", "attention_only",
      "--manifest", MANIFEST, "--tag", "attention_only_grouped"]),
    ("contrastive_only", "b_grouped_contrastive",
     ["src.training.train_variety", "--dataset", "b", "--experiment", "contrastive_only",
      "--manifest", MANIFEST, "--tag", "contrastive_only_grouped", "--encoder", ENCODER]),
    ("full (proposed)", "b_grouped_full",
     ["src.training.train_variety", "--dataset", "b", "--experiment", "full",
      "--manifest", MANIFEST, "--tag", "full_grouped", "--encoder", ENCODER]),
]

if __name__ == "__main__":
    os.makedirs("outputs/logs", exist_ok=True)
    log("=" * 60)
    log("Dataset B group-aware re-run: 89 train / 19 val / 19 test source seeds")
    for name, marker, cmd in STAGES:
        if not run(name, cmd, marker):
            sys.exit(1)
    log("ALL STAGES COMPLETE")
