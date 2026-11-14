"""표 모델 + 이미지 결합 후보 비교 (test_like val, train-only).

원본 DataTon/dataton_final 미수정. 작업은 ~/soomin/datathon_img 만.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score

import model as M
from pathlib import Path as _P
import sys

ROOT = Path(__file__).resolve().parent
SOOMIN = ROOT.parent
sys.path.insert(0, str(SOOMIN))
from common.seed import set_seed
from common.test_like import to_test_like

DATA = Path.home() / "data" / "battery_data_light"
ART = ROOT / "artifacts"
LABELS = list(range(1, 7))
WS = [0.05, 0.1, 0.15, 0.2]
MS = [0.1, 0.2, 0.3]
# 크기 불변 비율 피처 + 직전 대비 변화 (≤10)
COLOR_BASE = ["pt_frac50", "pt_frac70", "pt_frac90", "pt_p90", "pt_p99"]  # 5
# + 5 deltas → 10


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0))


def per_f1(y, p) -> list[float]:
    return [float(v) for v in f1_score(y, p, labels=LABELS, average=None, zero_division=0)]


def load_pool() -> pd.DataFrame:
    frames = []
    for split in ("train", "validation"):
        f = pd.read_csv(
            DATA / split / f"{split}.csv",
            dtype={"sample_id": "string", "experiment_id": "string"},
        )
        f["origin"] = split
        f["image_file"] = M.resolve_image_paths(f, [DATA / split, DATA]).astype(str).values
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def is_image_col(c: str) -> bool:
    return c.startswith("pt_") or c == "white_frac" or c.startswith("r_img")


def fuse_margin(p_tab: np.ndarray, p_cnn: np.ndarray, w: float, m: float) -> np.ndarray:
    """1·2위 확률 차이 < m 인 행만 융합, 나머지 P_tab."""
    order = np.argsort(-p_tab, axis=1)
    top = p_tab[np.arange(len(p_tab)), order[:, 0]]
    second = p_tab[np.arange(len(p_tab)), order[:, 1]]
    uncertain = (top - second) < m
    out = p_tab.copy()
    if uncertain.any():
        out[uncertain] = M.fuse(p_tab[uncertain], p_cnn[uncertain], w)
    return out


def color_delta_features(frame: pd.DataFrame, color: pd.DataFrame) -> pd.DataFrame:
    """비율 피처 5개 + 직전 프레임 대비 변화 5개 (실험·시간순, 과거만)."""
    c = color.reindex(frame["sample_id"].astype(str).values)
    base = c[COLOR_BASE].astype(float).to_numpy()
    n = len(frame)
    delta = np.zeros_like(base)
    eid = frame["experiment_id"].astype(str).to_numpy()
    sid = frame["sample_id"].astype(str).to_numpy()
    t = pd.to_numeric(frame["relative_time_s"], errors="coerce").to_numpy(dtype=float)
    for _, idx in pd.Series(np.arange(n)).groupby(eid, sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t[idx]))]
        x = base[idx]
        d = np.zeros_like(x)
        d[1:] = x[1:] - x[:-1]
        delta[idx] = d
    cols = {f"img10_{k}": base[:, i] for i, k in enumerate(COLOR_BASE)}
    cols.update({f"img10_d_{k}": delta[:, i] for i, k in enumerate(COLOR_BASE)})
    return pd.DataFrame(cols, index=pd.RangeIndex(n))


def predict_labels(p: np.ndarray, p23: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    return M.apply_rules(M.to_label(p), p23, frame)


def blank_change(pred_real: np.ndarray, pred_blank: np.ndarray) -> float:
    return float((np.asarray(pred_real) != np.asarray(pred_blank)).mean())


def make_blank_color(frame: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    """원본과 같은 크기의 검정 이미지로 색통계 재계산."""
    rows = []
    for p in paths:
        with Image.open(p) as im:
            size = im.size
            mode = im.mode
        tmp = ART / "_blank_tmp.png"
        Image.new("RGB", size, (0, 0, 0)).save(tmp)
        rows.append(M.color_stats(tmp))
    if tmp.exists():
        tmp.unlink()
    return pd.DataFrame(rows, columns=M.COLOR_STATS, index=pd.Index(frame["sample_id"].astype(str).values, name="sample_id"))


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    set_seed(42)
    t0 = time.time()
    rows_out = []

    pool = load_pool()
    color = joblib.load(ART / "color_features.joblib")
    # ensure index
    if color.index.name != "sample_id":
        if "sample_id" in color.columns:
            color = color.set_index("sample_id")
        color.index = color.index.astype(str)

    bundle_final = joblib.load(ROOT / "weights" / "tab_lgbm.joblib")
    columns = list(bundle_final["columns"])
    columns23 = list(bundle_final["columns23"])
    print(f"final bundle train_rows={bundle_final.get('train_rows')} n_exp={bundle_final.get('train_experiments')} fuse_w={bundle_final.get('fuse_w')}", flush=True)

    # --- train-only tab fit ---
    tr = (pool.origin == "train").to_numpy()
    print("build features…", flush=True)
    Xp = M.build_features(pool, color)
    y_tr = pool.loc[tr, "label"].to_numpy(int)
    print("fit train-only LightGBM…", flush=True)
    m = M.build_model().fit(Xp.loc[tr, columns], y_tr)
    er_tr = M.pressure_ranges(pool[tr])
    table_tr = M.range_table(er_tr)
    Q_tr = M.pressure_features(pool[tr], table_tr, own_range=M.loo_ranges(er_tr))
    m23 = M.build_model().fit(pd.concat([Xp.loc[tr, columns], Q_tr], axis=1)[columns23], y_tr)
    train_only_bundle = {
        "model": m,
        "columns": columns,
        "model23": m23,
        "columns23": columns23,
        "range_table": table_tr,
        "fuse_w": 0.0,
        "train_rows": int(tr.sum()),
        "train_experiments": int(pool.loc[tr, "experiment_id"].nunique()),
        "scope": "train-only",
    }
    joblib.dump(train_only_bundle, ART / "tab_train_only.joblib")
    print(f"train-only: rows={tr.sum()} exps={train_only_bundle['train_experiments']}", flush=True)

    # --- val test_like ---
    val = pool[pool.origin == "validation"].copy()
    ev = to_test_like(val)
    y = pd.to_numeric(ev["label"], errors="coerce").astype(int).to_numpy()
    print(f"test_like val rows={len(ev)}", flush=True)

    Xe = M.build_features(ev, color)[columns]
    p_tab = m.predict_proba(Xe)
    p_tab23 = m23.predict_proba(pd.concat([Xe, M.pressure_features(ev, table_tr)], axis=1)[columns23])

    # CNN train-only
    pack = joblib.load(ART / "val_testlike_proba_trainonly_cnn.joblib")
    # align by sample_id
    pack_ids = list(pack["sample_id"])
    ev_ids = ev["sample_id"].astype(str).tolist()
    if pack_ids == ev_ids:
        p_cnn = np.asarray(pack["p_cnn"], dtype=np.float64)
        print("CNN probs: reused artifacts/val_testlike_proba_trainonly_cnn.joblib (train-only CNN)", flush=True)
        cnn_source = "artifacts/val_testlike_proba_trainonly_cnn.joblib"
    else:
        print("sample_id mismatch — recompute CNN with cnn_train_only.pt", flush=True)
        M.set_torch_deterministic()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cnn = M.load_cnn(ART / "cnn_train_only.pt", device)
        images = M.decode_images(ev["image_file"].tolist())
        p_cnn = M.cnn_probs(ev, images, cnn, device)
        cnn_source = "artifacts/cnn_train_only.pt recomputed"
    # verify pack p_tab matches our fit (should be close/same if same seed)
    if pack_ids == ev_ids:
        dtab = float(np.max(np.abs(p_tab - np.asarray(pack["p_tab"]))))
        print(f"p_tab vs pack maxabs={dtab:.6g}", flush=True)

    # save real probs
    joblib.dump(
        {"sample_id": ev_ids, "p_tab": p_tab, "p_tab23": p_tab23, "p_cnn": p_cnn, "y": y, "cnn_source": cnn_source},
        ART / "val_testlike_real_proba.joblib",
    )

    # --- blank images: recompute color + CNN ---
    print("blank color + CNN…", flush=True)
    color_blank = make_blank_color(ev, ev["image_file"].tolist())
    Xe_b = M.build_features(ev, color_blank)[columns]
    p_tab_b = m.predict_proba(Xe_b)
    p_tab23_b = m23.predict_proba(pd.concat([Xe_b, M.pressure_features(ev, table_tr)], axis=1)[columns23])
    M.set_torch_deterministic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = M.load_cnn(ART / "cnn_train_only.pt", device)
    # decode blank images of matching sizes
    blank_paths = []
    blank_dir = ART / "blank_val_images"
    blank_dir.mkdir(exist_ok=True)
    for i, src in enumerate(ev["image_file"].tolist()):
        with Image.open(src) as im:
            size = im.size
        dst = blank_dir / f"{i}.png"
        Image.new("RGB", size, (0, 0, 0)).save(dst)
        blank_paths.append(str(dst))
    images_b = M.decode_images(blank_paths)
    p_cnn_b = M.cnn_probs(ev, images_b, cnn, device)
    joblib.dump(
        {"sample_id": ev_ids, "p_tab": p_tab_b, "p_tab23": p_tab23_b, "p_cnn": p_cnn_b},
        ART / "val_testlike_blank_proba.joblib",
    )

    # ========== baseline: tab only + v7 gates ==========
    # "표 모델" blank: color 재계산 없이 CNN만 비움 → 변화 0 기대. 실제 규정 빈이미지는 color도 바뀜 → 둘 다 기록.
    pred_base = predict_labels(p_tab, p_tab23, ev)
    pred_base_blank_tab = predict_labels(p_tab_b, p_tab23_b, ev)  # color 재계산
    pred_base_blank_cached = predict_labels(p_tab, p_tab23, ev)  # 캐시 color 유지(=이미지 미사용)
    f1_base = macro_f1(y, pred_base)
    blank_tab_recolor = blank_change(pred_base, pred_base_blank_tab)
    blank_tab_cached = blank_change(pred_base, pred_base_blank_cached)
    print(f"BASE tab+gates F1={f1_base:.6f} blank_recolor={blank_tab_recolor:.4%} blank_cached={blank_tab_cached:.4%}", flush=True)
    rows_out.append(
        {
            "candidate": "baseline_tab_gates",
            "w": 0.0,
            "m": "",
            "test_like_f1": f1_base,
            "delta_f1": 0.0,
            "blank_change_rate": blank_tab_recolor,
            "blank_change_cached_color": blank_tab_cached,
            "f1_1": per_f1(y, pred_base)[0],
            "f1_2": per_f1(y, pred_base)[1],
            "f1_3": per_f1(y, pred_base)[2],
            "f1_4": per_f1(y, pred_base)[3],
            "f1_5": per_f1(y, pred_base)[4],
            "f1_6": per_f1(y, pred_base)[5],
            "selected": False,
            "note": "train-only tab + v7 gates; blank_recolor=색통계 재계산",
        }
    )

    def eval_cand(name: str, p_real, p23_real, p_blank, p23_blank, w="", m="", note=""):
        pred = predict_labels(p_real, p23_real, ev)
        pred_b = predict_labels(p_blank, p23_blank, ev)
        f1 = macro_f1(y, pred)
        pf = per_f1(y, pred)
        br = blank_change(pred, pred_b)
        rec = {
            "candidate": name,
            "w": w,
            "m": m,
            "test_like_f1": f1,
            "delta_f1": f1 - f1_base,
            "blank_change_rate": br,
            "blank_change_cached_color": "",
            "f1_1": pf[0],
            "f1_2": pf[1],
            "f1_3": pf[2],
            "f1_4": pf[3],
            "f1_5": pf[4],
            "f1_6": pf[5],
            "selected": False,
            "note": note,
        }
        rows_out.append(rec)
        print(f"{name}: F1={f1:.6f} Δ={f1-f1_base:+.6f} blank={br:.4%}", flush=True)
        return rec

    # ========== (A) late fusion ==========
    for w in WS:
        p = M.fuse(p_tab, p_cnn, w)
        pb = M.fuse(p_tab_b, p_cnn_b, w)
        eval_cand(f"A_fuse_w{w}", p, p_tab23, pb, p_tab23_b, w=w, note="late geometric fuse")

    # ========== (B) margin-limited fusion ==========
    for w in WS:
        for m in MS:
            p = fuse_margin(p_tab, p_cnn, w, m)
            pb = fuse_margin(p_tab_b, p_cnn_b, w, m)
            eval_cand(f"B_margin_w{w}_m{m}", p, p_tab23, pb, p_tab23_b, w=w, m=m, note="fuse only if top1-top2 < m")

    # ========== (C) ≤10 color ratio+delta features, retrain ==========
    print("C: retrain with ≤10 img ratio/delta features…", flush=True)
    t_c0 = time.time()
    non_img = [c for c in columns if not is_image_col(c)]
    img10_pool = color_delta_features(pool, color)
    img10_ev = color_delta_features(ev, color)
    img10_names = list(img10_pool.columns)  # 10
    assert len(img10_names) <= 10
    cols_c = non_img + img10_names
    press_cols = [c for c in columns23 if c not in columns]
    cols23_c = cols_c + press_cols

    Xp_ni = Xp[non_img].copy()
    Xp_ni.index = pool.index
    img10_pool.index = pool.index
    Xp_c = pd.concat([Xp_ni, img10_pool], axis=1)[cols_c]
    m_c = M.build_model().fit(Xp_c.loc[tr, cols_c], y_tr)
    Q_tr_i = Q_tr.copy()
    Q_tr_i.index = pool.index[tr]
    m23_c = M.build_model().fit(pd.concat([Xp_c.loc[tr, cols_c], Q_tr_i], axis=1)[cols23_c], y_tr)

    Xe_ni = M.build_features(ev, color)[non_img].copy()
    Xe_ni.index = ev.index
    img10_ev.index = ev.index
    Xe_c = pd.concat([Xe_ni, img10_ev], axis=1)[cols_c]
    p_tab_c = m_c.predict_proba(Xe_c)
    Qe = M.pressure_features(ev, table_tr)
    Qe.index = ev.index
    p_tab23_c = m23_c.predict_proba(pd.concat([Xe_c, Qe], axis=1)[cols23_c])

    img10_ev_b = color_delta_features(ev, color_blank)
    img10_ev_b.index = ev.index
    Xe_ni_b = M.build_features(ev, color_blank)[non_img].copy()
    Xe_ni_b.index = ev.index
    Xe_c_b = pd.concat([Xe_ni_b, img10_ev_b], axis=1)[cols_c]
    p_tab_c_b = m_c.predict_proba(Xe_c_b)
    p_tab23_c_b = m23_c.predict_proba(pd.concat([Xe_c_b, Qe], axis=1)[cols23_c])
    # C alone (no CNN)
    eval_cand("C_img10_tab", p_tab_c, p_tab23_c, p_tab_c_b, p_tab23_c_b, note=f"retrain non_img+{len(img10_names)} color; {time.time()-t_c0:.1f}s")
    # C + small fuse
    for w in [0.05, 0.1]:
        p = M.fuse(p_tab_c, p_cnn, w)
        pb = M.fuse(p_tab_c_b, p_cnn_b, w)
        eval_cand(f"C_img10_fuse_w{w}", p, p_tab23_c, pb, p_tab23_c_b, w=w, note="C + late fuse")

    joblib.dump(
        {
            "model": m_c,
            "model23": m23_c,
            "columns": cols_c,
            "columns23": cols23_c,
            "range_table": table_tr,
            "img10_names": img10_names,
            "non_img": non_img,
            "press_cols": press_cols,
        },
        ART / "tab_train_only_img10.joblib",
    )

    # --- select ---
    df = pd.DataFrame(rows_out)
    # candidates only (exclude baseline)
    cand = df[df["candidate"] != "baseline_tab_gates"].copy()
    ok = cand[cand["blank_change_rate"] >= 0.01].copy()
    if len(ok) == 0:
        # fallback: max blank among cand
        ok = cand.copy()
        print("WARNING: no candidate with blank>=1%", flush=True)
    # minimize loss (= maximize delta_f1, i.e. least negative)
    ok = ok.sort_values(["delta_f1", "candidate"], ascending=[False, True])
    # among equal delta, prefer simpler: A over B over C, smaller w, no m
    def simplicity(r):
        name = r["candidate"]
        if name.startswith("A_"):
            return (0, float(r["w"] or 0), 0)
        if name.startswith("B_"):
            return (1, float(r["w"] or 0), float(r["m"] or 0))
        return (2, float(r["w"] or 0) if r["w"] != "" else 9, 0)

    best_delta = ok.iloc[0]["delta_f1"]
    tied = ok[np.isclose(ok["delta_f1"], best_delta, atol=1e-12)]
    if len(tied) > 1:
        tied = tied.copy()
        tied["_sim"] = tied.apply(simplicity, axis=1)
        tied = tied.sort_values("_sim")
        best = tied.iloc[0]
    else:
        best = ok.iloc[0]

    df.loc[df["candidate"] == best["candidate"], "selected"] = True
    metrics_path = ROOT / "metrics.csv"
    df.to_csv(metrics_path, index=False)

    meta = {
        "baseline_f1": f1_base,
        "baseline_blank_recolor": blank_tab_recolor,
        "baseline_blank_cached_color": blank_tab_cached,
        "selected": best["candidate"],
        "selected_delta": float(best["delta_f1"]),
        "selected_blank": float(best["blank_change_rate"]),
        "selected_f1": float(best["test_like_f1"]),
        "cnn_source": cnn_source,
        "train_only_exps": train_only_bundle["train_experiments"],
        "n_test_like": int(len(ev)),
        "elapsed_s": time.time() - t0,
        "img10_names": img10_names,
    }
    (ART / "selection.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("SELECTED", json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print("wrote", metrics_path, flush=True)


if __name__ == "__main__":
    main()
