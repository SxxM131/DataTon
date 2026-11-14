"""CNN 이미지 확률(OOF) + 센서 피처 → LightGBM 스태킹. 누수 없는 중첩 구조.

outer fold k:
  GBDT 학습 행(fold j≠k)의 이미지 확률 = pair(k,j) 모델(= fold k, j를 모두 안 본 CNN)의 pool 문맥 예측
  평가 행(fold k, evalp)의 이미지 확률   = out k 모델(fold k를 안 본 CNN)의 evalp 문맥 예측
추가 피처(opt): 이미지 확률의 인과적 집계 (30초 윈도우 평균, 현재까지 누적 최대, 기대 단계 E[stage])
이미지 의존성 점검: 평가 행의 이미지 확률을 같은 fold 안 다른 행의 것과 바꿨을 때 예측이 달라지는 비율.

예) python stack.py temporal
"""
import sys
import numpy as np
import pandas as pd
import imglib as L
from imglib import cv

tag = sys.argv[1] if len(sys.argv) > 1 else 'temporal'
pool, evalp = L.load_all()
N = len(pool)
y = pool.label.astype(int).to_numpy()
is_eval = np.zeros(N, bool); is_eval[pool.index.get_indexer(evalp.index)] = True
ev_rows = np.where(is_eval)[0]
Z = np.load(L.ICACHE / f'cnn_{tag}.npz')
Xs_fit, Xs_eval = L.sensor_features(pool, evalp)
color_cols = [c for c in Xs_fit.columns if c.startswith('pt_') or c == 'white_frac']
S_fit, S_eval = Xs_fit.drop(columns=color_cols).to_numpy(np.float32), Xs_eval.drop(columns=color_cols).to_numpy(np.float32)
folds = cv.make_folds(pool, n_repeats=1)
fold_of = np.full(N, -1)
for f in folds:
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
t = pool.relative_time_s.to_numpy(float)
exp_groups = list(pool.groupby('experiment_id', sort=False).indices.values())


def agg(P, mask):
    """P: (N,6) 확률 (mask 행만 유효). mask 행만 보이는 문맥에서 인과적 집계 → (N, F). 실험별·시간순."""
    out = np.full((N, 6 + 1 + 6 + 6 + 1), np.nan, dtype=np.float32)
    for idx in exp_groups:
        idx = idx[mask[idx]]
        if not len(idx):
            continue
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        p = P[idx]; tt = t[idx]
        es = p @ np.arange(1, 7)
        cs = np.cumsum(np.vstack([np.zeros((1, 6)), p]), 0)
        st = np.searchsorted(tt, tt - 30.0, side='right')
        m30 = (cs[np.arange(len(idx)) + 1] - cs[st]) / (np.arange(len(idx)) + 1 - st)[:, None]
        out[idx] = np.hstack([p, es[:, None], m30, np.maximum.accumulate(p, 0), np.maximum.accumulate(es)[:, None]])
    return out


def image_feats(k, mode):
    """(학습 행 피처 (N,F) — fold k 밖 행만 유효, 평가 행 피처 (N,F) — fold k의 evalp 행만 유효)"""
    Ptr = np.full((N, 6), np.nan, dtype=np.float32)
    for j in range(5):
        if j == k:
            continue
        src = Z[f'out{j}_pool'] if mode == 'naive' else Z[f'pair{min(k, j)}{max(k, j)}_pool']
        Ptr[fold_of == j] = src[fold_of == j]
    Pva = Z[f'out{k}_eval']
    return Ptr, Pva


def make_runner(use_sensor, use_agg, mode='nested', perm=False):
    def fp(tr, va, f):
        k = f['fold']
        Ptr, Pva = image_feats(k, mode)
        va_rows = ev_rows[va]
        if perm:                                           # 이미지 바꾸기: 같은 fold의 다른 행(무작위) 이미지 확률로 교체
            rng = np.random.default_rng(0)
            Pva = Pva.copy(); Pva[va_rows] = Pva[rng.permutation(va_rows)]
        if use_agg:
            Ftr, Fva = agg(Ptr, fold_of != k), agg(Pva, (fold_of == k) & is_eval)
        else:
            Ftr, Fva = Ptr, Pva
        A = np.hstack([S_fit[tr], Ftr[tr]]) if use_sensor else Ftr[tr]
        B = np.hstack([S_eval[va], Fva[va_rows]]) if use_sensor else Fva[va_rows]
        key = (k, use_sensor, use_agg, mode)
        if key not in models:
            models[key] = L.lgbm().fit(A, y[tr])
        return models[key].predict(B)
    return fp


models = {}
# 이미지 단독 (CNN argmax)
oof_img = pd.DataFrame({'pred_r0': np.concatenate([Z[f'out{k}_eval'][ev_rows[fold_of[ev_rows] == k]].argmax(1) + 1 for k in range(5)])},
                       index=np.concatenate([evalp.index[fold_of[ev_rows] == k] for k in range(5)])).reindex(evalp.index)
res = cv.evaluate(evalp, oof_img)
if 'official_eval' in Z:
    ov = (pool.origin == 'validation').to_numpy() & is_eval
    res['official_val'] = cv.macro_f1(y[ov], Z['official_eval'][ov].argmax(1) + 1)
cv.save_result(f'img_cnn_{tag}_only', res, oof_img); print(cv.summary_line(f'img_cnn_{tag}_only', res), flush=True)

have_pairs = 'pair34_pool' in Z
base = pd.read_pickle(L.HERE.parent / 'results/oof/img_tab_sensor_only.pkl')
ref = pd.read_pickle(L.REF_OOF)
out = {}
for name, kw in [
    (f'img_cnn_{tag}_agg_only', dict(use_sensor=False, use_agg=True)),
    (f'img_stack_{tag}', dict(use_sensor=True, use_agg=False)),
    (f'img_stack_{tag}_agg', dict(use_sensor=True, use_agg=True)),
    (f'img_stack_{tag}_agg_naive', dict(use_sensor=True, use_agg=True, mode='naive')),
]:
    if not have_pairs and kw.get('mode', 'nested') == 'nested':
        continue
    r, o = L.run_cv(name, pool, evalp, make_runner(**kw), official=False)
    out[name] = o
    L.cmp(evalp, o, base, name, 'lgbm sensor_only')
    if kw['use_sensor']:
        _, o2 = L.run_cv(name + '_PERM', pool, evalp, make_runner(**kw, perm=True), official=False, save=False, verbose=False)
        ch = (o2.pred_r0 != o.pred_r0).mean()
        print(f'    이미지 바꾸기: 예측이 달라진 비율 {ch:.3f} | 바꾼 뒤 macroF1 {cv.macro_f1(evalp.label, o2.pred_r0):.4f}', flush=True)
