"""이미지 트랙 공용 유틸 (공통 검증 체계 cv.py는 그대로 사용).

- 캐시된 이미지 배열 로드 (pool 행 순서와 정렬 확인)
- 인과적(과거 방향) 참조 프레임 인덱스: 실험 첫 프레임, W초 이전 프레임 (relative_time_s 기준)
  · 학습 행: pool(5초 간격 전체) 문맥,  평가 행: evalp(Test처럼 다운샘플된 행) 문맥에서 각각 계산
- 센서 피처(X_fit / X_eval) 캐시: exp02.make_X(color, extra) = reference와 동일 피처
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common'), str(HERE.parent), str(HERE.parent.parent)]
import cv  # noqa: E402

DATA = Path.home() / 'data/battery_data_light'
CACHE = Path.home() / 'jaehyeok/cache'
ICACHE = CACHE / 'img'
REF_OOF = HERE.parent / 'results/oof/ref_gbdt_color_extra.pkl'
NT = 4


def load_all():
    pool = cv.load_pool(DATA)
    evalp = cv.to_test_like(pool)
    return pool, evalp


def load_images(pool, kind='rgb', S=128, mmap=True):
    ids = np.load(ICACHE / 'pool_ids.npy')
    assert (ids == pool.sample_id.to_numpy().astype(str)).all(), '캐시와 pool 행 순서 불일치'
    return np.load(ICACHE / f'pool_{kind}{S}.npy', mmap_mode='r' if mmap else None)


def context_index(pool: pd.DataFrame, subset_mask: np.ndarray, lags=(30.0,)):
    """subset_mask로 고른 행들만 '보이는' 상황에서, 각 행의 참조 프레임 위치(pool 행 위치)를 돌려준다.

    first[i]  : 같은 실험에서 보이는 행 중 가장 이른 관측
    lag{W}[i] : 같은 실험에서 보이는 행 중 t <= t_i - W 인 마지막 관측 (없으면 first)
    prev[i]   : 바로 이전에 보이는 관측 (없으면 자기 자신)
    보이지 않는 행은 -1.
    """
    n = len(pool)
    t = pool['relative_time_s'].to_numpy(dtype=float)
    out = {'first': np.full(n, -1), 'prev': np.full(n, -1), **{f'lag{int(w)}': np.full(n, -1) for w in lags}}
    for _, idx in pool.groupby('experiment_id', sort=False).indices.items():
        idx = idx[subset_mask[idx]]
        if not len(idx):
            continue
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        tt = t[idx]
        out['first'][idx] = idx[0]
        out['prev'][idx] = np.r_[idx[0], idx[:-1]]
        for w in lags:
            j = np.searchsorted(tt, tt - w, side='right') - 1
            out[f'lag{int(w)}'][idx] = idx[np.maximum(j, 0)]
    return out


def sensor_features(pool, evalp, groups=('color', 'extra')):
    """reference(ref_gbdt)와 같은 피처. 학습용은 pool 전체에서, 평가용은 evalp 행만 보고 생성."""
    tag = '_'.join(groups) or 'base'
    f = ICACHE / f'sensorX_{tag}.pkl'
    if f.exists():
        return pd.read_pickle(f)
    from exp02_gbdt import make_X
    color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
    X_fit, X_eval = make_X(pool, color, list(groups)), make_X(evalp, color, list(groups))
    pd.to_pickle((X_fit, X_eval), f)
    return X_fit, X_eval


def lgbm(n_estimators=300, seed=42, **kw):
    import lightgbm as lgb
    p = dict(objective='multiclass', learning_rate=0.05, n_estimators=n_estimators, num_leaves=31, min_child_samples=50,
             reg_lambda=1.0, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, n_jobs=NT, random_state=seed, verbose=-1)
    p.update(kw)
    return lgb.LGBMClassifier(**p)


def run_cv(name, pool, evalp, fit_predict, n_repeats=1, official=True, save=True, verbose=True):
    """fit_predict(train_mask_over_pool, valid_mask_over_evalp, fold) -> 예측 라벨(1~6) 배열 (evalp[valid] 순서)."""
    oof = pd.DataFrame(index=evalp.index)
    for f in cv.make_folds(pool, n_repeats=n_repeats):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
        va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        oof.loc[evalp.index[va], f"pred_r{f['rep']}"] = fit_predict(tr, va, f)
    oof = oof.astype(int)
    res = cv.evaluate(evalp, oof)
    if official:
        f = cv.official_split(pool)
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
        va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        res['official_val'] = cv.macro_f1(evalp.loc[va, 'label'], fit_predict(tr, va, f))
    if save:
        cv.save_result(name, res, oof)
    if verbose:
        print(cv.summary_line(name, res), flush=True)
        if REF_OOF.exists():
            c = cv.compare(evalp, oof, pd.read_pickle(REF_OOF))
            print(f"    vs ref_gbdt_color_extra: diff {c['diff']:+.4f} ci90 [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p_better {c['p_a_better']:.3f}", flush=True)
    return res, oof


def cmp(evalp, a, b, la='A', lb='B'):
    c = cv.compare(evalp, a, b)
    print(f"    {la} vs {lb}: diff {c['diff']:+.4f} ci90 [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p_better {c['p_a_better']:.3f}", flush=True)
    return c
