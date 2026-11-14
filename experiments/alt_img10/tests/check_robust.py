"""datathon_img/test.py 견고성 스모크 (소량 행)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / "data" / "battery_data_light"
PY = "/home/ubuntu/venv/bin/python"


def main() -> int:
    raw = pd.read_csv(DATA / "test" / "test.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    exps = raw["experiment_id"].drop_duplicates().head(2).tolist()
    parts = [raw[raw.experiment_id == e].sort_values("relative_time_s").head(30) for e in exps]
    frame = pd.concat(parts, ignore_index=True)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inp, out = tmp / "in.csv", tmp / "out.csv"
        frame.to_csv(inp, index=False)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2"}
        r = subprocess.run(
            [PY, str(REPO / "test.py"), "--input", str(inp), "--image_root", str(DATA / "test"),
             "--output", str(out), "--device", "cuda"],
            cwd=str(REPO), capture_output=True, text=True, env=env,
        )
        ok = r.returncode == 0 and out.exists()
        msg = ""
        if ok:
            pred = pd.read_csv(out)
            ok = set(pred.columns) == {"sample_id", "label"} and pred.label.between(1, 6).all() and len(pred) == len(frame)
            msg = f"n={len(pred)} labels={pred.label.value_counts().to_dict()}"
        else:
            msg = r.stderr[-500:]
        results.append({"case": "small_2exp", "ok": bool(ok), "msg": msg})
        # shuffled
        if ok:
            sh = frame.sample(frac=1, random_state=0).reset_index(drop=True)
            sh.to_csv(inp, index=False)
            out2 = tmp / "out2.csv"
            r2 = subprocess.run(
                [PY, str(REPO / "test.py"), "--input", str(inp), "--image_root", str(DATA / "test"),
                 "--output", str(out2), "--device", "cuda"],
                cwd=str(REPO), capture_output=True, text=True, env=env,
            )
            p1 = pd.read_csv(out).set_index("sample_id")["label"]
            p2 = pd.read_csv(out2).set_index("sample_id")["label"]
            same = bool((p1.loc[p2.index] == p2).all())
            results.append({"case": "shuffle_consistent", "ok": same, "msg": f"same={same}"})

    all_ok = all(x["ok"] for x in results)
    path = REPO / "logs" / "check_robust.json"
    path.write_text(json.dumps({"all_ok": all_ok, "results": results}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"all_ok": all_ok, "results": results}, ensure_ascii=False, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
