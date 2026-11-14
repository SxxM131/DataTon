"""재현·규정 4종: shuffle / 실험분리 / 미래절단 / 빈 이미지 (datathon_img/test.py)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / "data" / "battery_data_light"
TEST_CSV = DATA / "test" / "test.csv"
IMAGE_ROOT = DATA / "test"
PY = Path("/home/ubuntu/venv/bin/python")


def run_predict(frame: pd.DataFrame, image_root: Path, work: Path, tag: str) -> pd.Series:
    inp, out = work / f"{tag}_in.csv", work / f"{tag}_out.csv"
    frame.to_csv(inp, index=False)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2"}
    r = subprocess.run(
        [str(PY), str(REPO / "test.py"), "--input", str(inp), "--image_root", str(image_root),
         "--output", str(out), "--device", "cuda"],
        cwd=str(REPO), capture_output=True, text=True, env=env,
    )
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"[{tag}] fail\n{r.stderr[-2000:]}\n{r.stdout[-1000:]}")
    return pd.read_csv(out, dtype={"sample_id": "string"}).set_index("sample_id")["label"].astype(int)


def same_pred(a: pd.Series, b: pd.Series) -> bool:
    common = a.index.intersection(b.index)
    return bool((a.loc[common].to_numpy() == b.loc[common].to_numpy()).all()) and len(common) == len(a) == len(b)


def main() -> int:
    frame = pd.read_csv(TEST_CSV, dtype={"sample_id": "string", "experiment_id": "string"})
    report: dict = {"n_rows": int(len(frame))}
    with tempfile.TemporaryDirectory(prefix="img_check_repro_") as tmp:
        work = Path(tmp)
        baseline = run_predict(frame, IMAGE_ROOT, work, "base")
        report["label_dist"] = {int(k): int(v) for k, v in baseline.value_counts().sort_index().items()}

        shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        report["shuffle_same"] = same_pred(baseline, run_predict(shuffled, IMAGE_ROOT, work, "shuf").loc[baseline.index])

        parts = [run_predict(sub.reset_index(drop=True), IMAGE_ROOT, work, f"s{i}")
                 for i, (_, sub) in enumerate(frame.groupby("experiment_id", sort=False))]
        report["split_by_exp_same"] = same_pred(baseline, pd.concat(parts).loc[baseline.index])

        keep = []
        for _, sub in frame.groupby("experiment_id", sort=False):
            sub = sub.sort_values("relative_time_s", kind="mergesort")
            keep.append(sub.iloc[: max(1, int(np.ceil(len(sub) * 0.8)))])
        trunc = pd.concat(keep).reset_index(drop=True)
        pred_t = run_predict(trunc, IMAGE_ROOT, work, "trunc")
        common = baseline.index.intersection(pred_t.index)
        report["truncate_future_same"] = bool((baseline.loc[common].to_numpy() == pred_t.loc[common].to_numpy()).all())
        report["truncate_n"] = int(len(common))

        blank_root = work / "blank"
        made = {}
        for rel in frame["image_path"].astype(str).unique():
            src = IMAGE_ROOT / rel
            with Image.open(src) as im:
                key = (im.mode, im.size)
            dst = blank_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if key not in made:
                p = work / f"b{len(made)}.png"
                Image.new(key[0], key[1], 0).save(p)
                made[key] = p
            Image.open(made[key]).save(dst)
        pred_b = run_predict(frame, blank_root, work, "blank")
        rate = float((baseline.loc[pred_b.index].to_numpy() != pred_b.to_numpy()).mean())
        report["blank_image_change_rate"] = rate
        report["blank_image_changes"] = bool(rate > 0)

    report["all_pass"] = bool(
        report["shuffle_same"] and report["split_by_exp_same"]
        and report["truncate_future_same"] and report["blank_image_changes"]
    )
    out = REPO / "logs" / "check_repro.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
