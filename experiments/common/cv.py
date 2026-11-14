"""공통 검증 체계 (모든 실험 트랙이 같은 기준으로 비교하기 위한 모듈).

설계 근거
- Test는 실험 11개뿐이고, 500행이 넘는 실험은 500행으로 균등 간격 다운샘플되어 있다
  (floor(linspace(0, N-1, 500)) 규칙이 Test의 10초/15초 간격 패턴과 정확히 일치).
  반면 Train/Validation은 긴 실험이 원본 5초 간격 그대로라 행 기준 점수의 53~66%를 긴 실험 몇 개가 차지한다.
  → 평가 행은 반드시 Test와 같은 방식으로 변환(to_test_like)한 뒤 채점한다.
- 제공 Validation은 실험 19개라 점수 흔들림이 크다.
  → Train+Validation 97개 실험을 합쳐 실험 단위 층화 5-fold × 반복으로 OOF 점수를 낸다.
- 모델 간 차이는 실험 단위 paired bootstrap으로 유의한지 본다.

사용법
    pool  = load_pool(data_dir)          # 학습에 쓸 수 있는 전체 행 (97개 실험)
    evalp = to_test_like(pool)           # 채점 대상 행 (Test와 같은 500행 상한). pool의 부분집합, 인덱스 유지
    for f in make_folds(pool):           # f: rep, fold, train_exps, valid_exps
        학습: pool[pool.experiment_id.isin(f['train_exps'])]
        예측: evalp[evalp.experiment_id.isin(f['valid_exps'])]  ← 피처는 evalp 행만 보고 만든다 (Test에서 보이는 행과 동일)
    oof: DataFrame(index=evalp.index, columns=['pred_r0', 'pred_r1', ...])  (라벨 1~6)
    res = evaluate(evalp, oof); save_result(name, res, oof)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

LABELS = [1, 2, 3, 4, 5, 6]
CAP = 500                                   # Test의 실험당 최대 행 수
TEST_BAT_CLASS_COUNTS = {1: 5, 2: 2, 3: 4}  # Test 11개 실험의 bat_class 구성 (메타 컬럼, 라벨 아님)
OUT_DIR = Path(__file__).resolve().parent.parent / 'results'


def load_pool(data_dir) -> pd.DataFrame:
    frames = []
    for split in ('train', 'validation'):
        f = pd.read_csv(Path(data_dir) / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        f['origin'] = split
        f['image_file'] = str(Path(data_dir) / split) + '/' + f['image_path'].astype(str)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def to_test_like(frame: pd.DataFrame, cap: int = CAP) -> pd.DataFrame:
    """실험별로 시간순 정렬 후, cap행을 넘으면 Test와 같은 규칙으로 다운샘플. 원래 인덱스를 유지한 부분집합을 돌려준다."""
    keep = np.zeros(len(frame), dtype=bool)
    t = frame['relative_time_s'].to_numpy()
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        if len(idx) > cap:
            idx = idx[np.floor(np.linspace(0, len(idx) - 1, cap)).astype(int)]
        keep[idx] = True
    return frame[keep]


def exp_table(pool: pd.DataFrame) -> pd.DataFrame:
    g = pool.groupby('experiment_id', sort=True)
    et = g[['origin', 'bat_class', 'bat_cc', 'soc_status', 'tr_method']].first()
    et['n_rows'] = g.size()
    et['long'] = et['n_rows'] > CAP
    return et


def make_folds(pool: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3, seed: int = 42) -> list[dict]:
    """실험 단위 층화(bat_class × tr_method) K-fold. 실험 ID 정렬 + 시드 고정이라 어디서 불러도 같은 분할."""
    et = exp_table(pool)
    strata = (et['bat_class'] * 10 + et['tr_method']).to_numpy()
    exps = et.index.to_numpy()
    folds = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for k, (a, b) in enumerate(skf.split(exps, strata)):
            folds.append({'rep': rep, 'fold': k, 'train_exps': exps[a], 'valid_exps': exps[b]})
    return folds


def official_split(pool: pd.DataFrame) -> dict:
    """대회 제공 분할 (Train 78 → Validation 19). README에 적을 보조 지표용."""
    et = exp_table(pool)
    return {'rep': 'official', 'fold': 0, 'train_exps': et.index[et.origin == 'train'].to_numpy(),
            'valid_exps': et.index[et.origin == 'validation'].to_numpy()}


# ---------------------------------------------------------------------------
# 채점
# ---------------------------------------------------------------------------
def _f1_from_conf(c: np.ndarray) -> np.ndarray:
    """c[..., 정답, 예측] → 클래스별 F1 (분모 0이면 0, sklearn zero_division=0과 동일)."""
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    den = c.sum(-1) + c.sum(-2)
    return np.divide(2 * tp, den, out=np.zeros(den.shape, dtype=float), where=den > 0)


def macro_f1(y, p) -> float:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    c = np.zeros((6, 6))
    np.add.at(c, (y - 1, p - 1), 1)
    return float(_f1_from_conf(c).mean())


def _conf_by_exp(evalp: pd.DataFrame, pred: np.ndarray):
    exps, codes = np.unique(evalp['experiment_id'].to_numpy().astype(str), return_inverse=True)
    c = np.zeros((len(exps), 6, 6))
    np.add.at(c, (codes, evalp['label'].to_numpy(dtype=int) - 1, np.asarray(pred, dtype=int) - 1), 1)
    return exps, c


def evaluate(evalp: pd.DataFrame, oof: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """oof: index=evalp.index, 열 pred_r* (반복별 예측 라벨). 반복 평균 점수와 불확실성을 돌려준다."""
    oof = oof.reindex(evalp.index)
    cols = [c for c in oof.columns if c.startswith('pred_r')]
    assert cols and oof[cols].notna().all().all(), 'evalp의 모든 행에 예측이 있어야 합니다.'
    et = exp_table(evalp).reindex(np.unique(evalp['experiment_id'].to_numpy().astype(str)))
    n_rows_full = et['n_rows'].to_numpy()
    conf = np.mean([_conf_by_exp(evalp, oof[c].to_numpy())[1] for c in cols], axis=0)   # (실험, 6, 6) 반복 평균
    per_rep = [macro_f1(evalp['label'], oof[c]) for c in cols]
    res = {
        'cv': float(np.mean(per_rep)), 'cv_per_rep': [round(v, 4) for v in per_rep],
        'per_class': _f1_from_conf(conf.sum(0)).round(4).tolist(),
    }
    for key, mask in [('origin_train', (et.origin == 'train').to_numpy()), ('origin_val', (et.origin == 'validation').to_numpy()),
                      ('long_exps', n_rows_full >= CAP), ('short_exps', n_rows_full < CAP),
                      ('bat1', (et.bat_class == 1).to_numpy()), ('bat2', (et.bat_class == 2).to_numpy()), ('bat3', (et.bat_class == 3).to_numpy())]:
        res[key] = float(_f1_from_conf(conf[mask].sum(0)).mean())
    rng = np.random.default_rng(seed)
    n = len(conf)
    boot = np.array([_f1_from_conf(conf[rng.integers(0, n, n)].sum(0)).mean() for _ in range(n_boot)])
    res['boot_ci90'] = [float(np.quantile(boot, .05)), float(np.quantile(boot, .95))]
    # Test 크기(11개 실험, bat_class 구성 동일)로 뽑았을 때 점수 분포와 Public(행 30%) - Private(70%) 차이
    groups = {b: np.where((et.bat_class == b).to_numpy())[0] for b in TEST_BAT_CLASS_COUNTS}
    sim, gap = [], []
    for _ in range(n_boot):
        pick = np.concatenate([rng.choice(groups[b], k, replace=False) for b, k in TEST_BAT_CLASS_COUNTS.items()])
        c = np.rint(conf[pick].sum(0)).astype(int)
        pub = rng.binomial(c, 0.3)
        sim.append(_f1_from_conf(c).mean())
        gap.append(_f1_from_conf(pub).mean() - _f1_from_conf(c - pub).mean())
    res['test11_mean'], res['test11_sd'] = float(np.mean(sim)), float(np.std(sim))
    res['test11_q05_q95'] = [float(np.quantile(sim, .05)), float(np.quantile(sim, .95))]
    res['public_minus_private_sd'] = float(np.std(gap))
    return res


def compare(evalp: pd.DataFrame, oof_a: pd.DataFrame, oof_b: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> dict:
    """실험 단위 paired bootstrap. diff = A - B. 공통 반복만 사용."""
    cols = sorted(set(c for c in oof_a.columns if c.startswith('pred_r')) & set(oof_b.columns))
    ca = np.mean([_conf_by_exp(evalp, oof_a.reindex(evalp.index)[c].to_numpy())[1] for c in cols], axis=0)
    cb = np.mean([_conf_by_exp(evalp, oof_b.reindex(evalp.index)[c].to_numpy())[1] for c in cols], axis=0)
    rng = np.random.default_rng(seed)
    n = len(ca)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        d.append(_f1_from_conf(ca[i].sum(0)).mean() - _f1_from_conf(cb[i].sum(0)).mean())
    d = np.array(d)
    return {'diff': float(_f1_from_conf(ca.sum(0)).mean() - _f1_from_conf(cb.sum(0)).mean()),
            'ci90': [float(np.quantile(d, .05)), float(np.quantile(d, .95))], 'p_a_better': float((d > 0).mean()), 'repeats': cols}


def summary_line(name: str, res: dict) -> str:
    return (f"{name:34s} CV {res['cv']:.4f} [{res['boot_ci90'][0]:.3f},{res['boot_ci90'][1]:.3f}] | "
            f"train-origin {res['origin_train']:.3f} val-origin {res['origin_val']:.3f} | long {res['long_exps']:.3f} short {res['short_exps']:.3f} | "
            f"bat1/2/3 {res['bat1']:.3f}/{res['bat2']:.3f}/{res['bat3']:.3f} | per-class {np.round(res['per_class'], 3).tolist()}"
            + (f" | official-val {res['official_val']:.4f}" if 'official_val' in res else ''))


def save_result(name: str, res: dict, oof: pd.DataFrame | None = None, extra: dict | None = None):
    """results/<name>.json (커밋 대상) + results/oof/<name>.pkl (커밋 금지, .gitignore)."""
    (OUT_DIR / 'oof').mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f'{name}.json').write_text(json.dumps({'name': name, **res, **(extra or {})}, ensure_ascii=False, indent=1))
    if oof is not None:
        oof.to_pickle(OUT_DIR / 'oof' / f'{name}.pkl')
