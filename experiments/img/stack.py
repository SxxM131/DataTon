"""CNN 이미지 확률(OOF) + 센서 피처 → LightGBM 스태킹. 누수 없는 중첩 구조.

outer fold k:
  GBDT 학습 행(fold j≠k)의 이미지 확률 = pair(k,j) 모델(= fold k, j를 모두 안 본 CNN)의 pool 문맥 예측
  평가 행(fold k, evalp)의 이미지 확률   = out k 모델(fold k를 안 본 CNN)의 evalp 문맥 예측
추가 피처(agg): 이미지 확률의 인과적 집계 (30초 윈도우 평균, 현재까지 누적 최대, 기대 단계 E[stage])
이미지 의존성 점검: 평가 행의 이미지 확률을 같은 fold 안 다른 행(무작위)의 것과 바꿨을 때 예측이 달라지는 비율.

--base ref : reference 피처(exp02 make_X color,extra)에서 색 통계를 뺀 센서 피처 + imglib.lgbm
--base tab : 세션 A의 experiments/tab/feats.build(extra=True) 323개 피처 + tab/run.py의 LightGBM 설정 (n_jobs=4)

예) python stack.py --tag temporal --base ref
"""
import argparse
import sys
import numpy as np
import pandas as pd
import imglib as L
from imglib import cv

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='temporal')
ap.add_argument('--base', default='ref')
ap.add_argument('--variants', default='probs,agg')
ap.add_argument('--naive', action='store_true')
args = ap.parse_args()
tag = args.tag
pool, evalp = L.load_all()
N = len(pool)
y = pool.label.astype(int).to_numpy()
is_eval = np.zeros(N, bool); is_eval[pool.index.get_indexer(evalp.index)] = True
ev_rows = np.where(is_eval)[0]
class _Z:                                                   # 접근할 때마다 파일을 다시 연다 (학습 중 갱신 대비)
    def __getitem__(self, k):
        with np.load(L.ICACHE / f'cnn_{tag}.npz') as z:
            return z[k]
Z = _Z()
if args.base == 'ref':
    Xs_fit, Xs_eval = L.sensor_features(pool, evalp)
    color_cols = [c for c in Xs_fit.columns if c.startswith('pt_') or c == 'white_frac']
    S_fit, S_eval = Xs_fit.drop(columns=color_cols).to_numpy(np.float32), Xs_eval.drop(columns=color_cols).to_numpy(np.float32)
    new_model = lambda: L.lgbm()
else:
    f = L.ICACHE / 'tabX_all.pkl'
    if f.exists():
        S_fit, S_eval = pd.read_pickle(f)
    else:
        sys.path.insert(0, str(L.HERE.parent / 'tab'))
        import feats
        color = pd.concat([pd.read_pickle(L.CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
        S_fit, S_eval = feats.build(pool, color, {'extra': True}), feats.build(evalp, color, {'extra': True})
        pd.to_pickle((S_fit, S_eval), f)
    S_fit, S_eval = S_fit.to_numpy(np.float32), S_eval.to_numpy(np.float32)
    import lightgbm as lgb
    new_model = lambda: lgb.LGBMClassifier(objective='multiclass', learning_rate=0.05, n_estimators=400, num_leaves=31, min_child_samples=40,
                                           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, n_jobs=L.NT, verbose=-1,
                                           deterministic=True, force_row_wise=True, random_state=42)
print(f'base={args.base} 센서 피처 {S_fit.shape[1]}개, CNN tag={tag}', flush=True)
folds = cv.make_folds(pool, n_repeats=1)
fold_of = np.full(N, -1)
for f in folds:
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
t = pool.relative_time_s.to_numpy(float)
exp_groups = list(pool.groupby('experiment_id', sort=False).indices.values())


def agg(P, mask):
    """P: (N,6) 확률 (mask 행만 유효). mask 행만 보이는 문맥에서 인과적 집계 → (N, 20). 실험별·시간순."""
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
    Ptr = np.full((N, 6), np.nan, dtype=np.float32)
    for j in range(5):
        if j == k:
            continue
        src = Z[f'out{j}_pool'] if mode == 'naive' else Z[f'pair{min(k, j)}{max(k, j)}_pool']
        Ptr[fold_of == j] = src[fold_of == j]
    return Ptr, Z[f'out{k}_eval']


models = {}


def make_runner(use_sensor=True, use_img=True, use_agg=False, mode='nested', perm=False):
    def fp(tr, va, f):
        k = f['fold']
        va_rows = ev_rows[va]
        if not use_img:
            A, B = S_fit[tr], S_eval[va]
        else:
            Ptr, Pva = image_feats(k, mode)
            if perm:                                       # 이미지 바꾸기: 같은 fold의 다른 행(무작위) 이미지 확률로 교체
                rng = np.random.default_rng(0)
                Pva = Pva.copy(); Pva[va_rows] = Pva[rng.permutation(va_rows)]
            Ftr, Fva = (agg(Ptr, fold_of != k), agg(Pva, (fold_of == k) & is_eval)) if use_agg else (Ptr, Pva)
            A = np.hstack([S_fit[tr], Ftr[tr]]) if use_sensor else Ftr[tr]
            B = np.hstack([S_eval[va], Fva[va_rows]]) if use_sensor else Fva[va_rows]
        key = (k, use_sensor, use_img, use_agg, mode)
        if key not in models:
            models[key] = (new_model() if use_sensor else L.lgbm()).fit(A, y[tr])
        return models[key].predict(B)
    return fp


def conf23(o):
    yy, p = evalp.label.to_numpy(), o.pred_r0.to_numpy()
    return f"2→3 오분류 {((yy == 2) & (p == 3)).sum()}, 3→2 {((yy == 3) & (p == 2)).sum()}, 4↔5 {((yy == 4) & (p == 5)).sum() + ((yy == 5) & (p == 4)).sum()}"


# 이미지 단독 (CNN argmax)
oof_img = pd.DataFrame({'pred_r0': np.concatenate([Z[f'out{k}_eval'][ev_rows[fold_of[ev_rows] == k]].argmax(1) + 1 for k in range(5)])},
                       index=np.concatenate([evalp.index[fold_of[ev_rows] == k] for k in range(5)])).reindex(evalp.index)
res = cv.evaluate(evalp, oof_img)
ov = (pool.origin == 'validation').to_numpy() & is_eval
res['official_val'] = cv.macro_f1(y[ov], Z['official_eval'][ov].argmax(1) + 1)
cv.save_result(f'img_cnn_{tag}_only', res, oof_img); print(cv.summary_line(f'img_cnn_{tag}_only', res), flush=True)

bname = f'img_base_{args.base}'
bp = L.HERE.parent / f'results/oof/{bname}.pkl'
if bp.exists():
    base = pd.read_pickle(bp)
    print(cv.summary_line(bname, cv.evaluate(evalp, base)), flush=True)
else:
    _, base = L.run_cv(bname, pool, evalp, make_runner(use_img=False), official=False)
print('   ', conf23(base), flush=True)
V = {'img_agg_only': dict(use_sensor=False, use_agg=True), 'probs': dict(), 'agg': dict(use_agg=True)}
todo = [(f'img_cnn_{tag}_agg_only', V['img_agg_only'])] if args.base == 'ref' else []
todo += [(f'img_stack_{args.base}_{tag}_{v}', V[v]) for v in args.variants.split(',')]
if args.naive:
    todo.append((f'img_stack_{args.base}_{tag}_agg_naive', dict(use_agg=True, mode='naive')))
for name, kw in todo:
    r, o = L.run_cv(name, pool, evalp, make_runner(**kw), official=False)
    c = L.cmp(evalp, o, base, name, bname)
    print('   ', conf23(o), flush=True)
    if kw.get('use_sensor', True):
        _, o2 = L.run_cv(name + '_PERM', pool, evalp, make_runner(**kw, perm=True), official=False, save=False, verbose=False)
        ch = (o2.pred_r0 != o.pred_r0).mean()
        print(f'    이미지 바꾸기: 예측이 달라진 비율 {ch:.3f} | 바꾼 뒤 macroF1 {cv.macro_f1(evalp.label, o2.pred_r0):.4f}', flush=True)
