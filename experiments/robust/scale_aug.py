"""압력 배율 증강: 학습 실험의 압력에 임의 배율(1/3~3배, log 균등)을 곱한 복사본을 추가해 학습 (라벨 불변: 규칙이 실험별 min-max 정규화 기준).

평가: 공통 검증 rep 0. 검증 실험의 압력을 그대로 / ×0.3 / ×3 으로 바꿔 채점 (perturb.py와 같은 방식).
비교 기준 (증강 없음, perturb.py): 원본 0.9057, ×0.3 0.8897, ×3 0.8959 (2·3 재판정 포함, 게이트 제외).
"""
import argparse, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=2); ap.add_argument('--lo', type=float, default=1 / 3); ap.add_argument('--hi', type=float, default=3.0)
ap.add_argument('--only_b', action='store_true'); ap.add_argument('--tag', default='k2')
args = ap.parse_args()
C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
PR = {**M.LGB_PARAMS, 'n_jobs': 6}
rng = np.random.default_rng(42)
exps = pool.experiment_id.unique()
scale = {k: dict(zip(exps, np.exp(rng.uniform(np.log(args.lo), np.log(args.hi), len(exps))))) for k in range(args.K)}


def build(fr, tab, loo=None):
    col = color.reindex(fr['src_id'].values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
    X = M.build_features(fr, col)
    own = None if loo is None else {e: loo[o] for e, o in zip(fr.experiment_id, fr['src_exp'])}
    return X, pd.concat([X, M.pressure_features(fr, tab, own_range=own)], axis=1)


def with_src(fr):
    return fr.assign(src_id=fr.sample_id.astype(str), src_exp=fr.experiment_id.astype(str))


ye = evalp.label.to_numpy(int)
PERT = [('원본', 1.0), ('×0.3', 0.3), ('×3', 3.0)]
pred = {n: {'base': np.zeros(len(evalp), int), 'aug': np.zeros(len(evalp), int)} for n, _ in PERT}
for f in cv.make_folds(pool, n_repeats=1):
    trf = with_src(pool[pool.experiment_id.isin(f['train_exps'])]); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    er = M.pressure_ranges(trf); tab = M.range_table(er); loo = M.loo_ranges(er)
    parts = [trf]
    for k in range(args.K):
        c = trf.copy(); c['meas_p'] = c.meas_p * c.experiment_id.map(scale[k]).to_numpy()
        c['sample_id'] = c.sample_id.astype(str) + f'@{k}'; c['experiment_id'] = c.experiment_id.astype(str) + f'@{k}'; parts.append(c)
    Xa0, Xb0 = build(trf, tab, loo)
    aug = pd.concat(parts, ignore_index=True)
    Xa1, Xb1 = build(aug, tab, loo)
    y0, y1 = trf.label.to_numpy(int), aug.label.to_numpy(int)
    a0 = lgb.LGBMClassifier(**PR).fit(Xa0, y0); b0 = lgb.LGBMClassifier(**PR).fit(Xb0, y0)
    a1 = a0 if args.only_b else lgb.LGBMClassifier(**PR).fit(Xa1, y1); b1 = lgb.LGBMClassifier(**PR).fit(Xb1, y1)
    for n, s in PERT:
        ev = with_src(evalp[va].assign(meas_p=evalp.meas_p[va] * s))
        Xe, Xq = build(ev, tab)
        pred[n]['base'][va] = M.redecide_23(a0.predict(Xe), b0.predict_proba(Xq))
        pred[n]['aug'][va] = M.redecide_23(a1.predict(Xe), b1.predict_proba(Xq))
    print('fold', f['fold'], {n: (round(cv.macro_f1(ye[va], pred[n]['base'][va]), 3), round(cv.macro_f1(ye[va], pred[n]['aug'][va]), 3)) for n, _ in PERT}, flush=True)
from sklearn.metrics import f1_score
for n, _ in PERT:
    for k in ('base', 'aug'):
        p = pred[n][k]
        print(f"{n:5s} {'증강 없음' if k == 'base' else '배율 증강':6s} CV {cv.macro_f1(ye, p):.4f} {np.round(f1_score(ye, p, labels=range(1, 7), average=None), 3).tolist()}", flush=True)
    b = pd.DataFrame({'pred_r0': pred[n]['base']}, index=evalp.index); a = pd.DataFrame({'pred_r0': pred[n]['aug']}, index=evalp.index)
    c = cv.compare(evalp, a, b, n_boot=3000); print(f"   증강 − 기준: {c['diff']:+.4f} [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p {c['p_a_better']:.2f}", flush=True)
pd.to_pickle(pred, C / f'robust_scale_aug_{args.tag}.pkl')
