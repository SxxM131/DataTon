"""선형(베이스라인 LR)과 GBDT 확률을 섞으면 더 안정적인지 확인 + 진단용 제출 파일 생성.

배경: Public LB에서 GBDT(0.832, 0.825)가 LR(0.852)보다 낮게 나옴. CV에서는 GBDT가 0.07 높음.
- CV(rep 0): LR OOF 확률(베이스라인 피처) × LightGBM OOF 확률(기준 피처, seq/make_proba.py 캐시) 가중 평균
- --submit: 97개 실험 전체로 두 모델을 학습해 Test 확률을 섞은 제출 파일 저장 (진단용)

예) OMP_NUM_THREADS=4 python experiments/common/blend_lr_gbdt.py --submit
"""
import argparse
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / 'seq'), str(HERE.parent.parent)]
import cv  # noqa: E402
import model as M  # noqa: E402
from decode import decode, fit_params  # noqa: E402
from exp02_gbdt import make_X  # noqa: E402

CACHE = Path.home() / 'jaehyeok' / 'cache'
ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', type=Path, default=Path.home() / 'data' / 'battery_data_light')
ap.add_argument('--submit', action='store_true')
args = ap.parse_args()

pool = cv.load_pool(args.data_dir)
evalp = cv.to_test_like(pool)
y = pool['label'].astype(int)
gray = pd.concat([pd.read_pickle(CACHE / f'{s}_gray.pkl') for s in ('train', 'validation')])
XL_fit, XL_eval = M.build_features(pool, gray), M.build_features(evalp, gray)
d = pickle.load(open(CACHE / 'seq' / 'proba_outer.pkl', 'rb'))
assert (d['index'] == evalp.index.to_numpy()).all()
PG = d['proba'][0]
PL = np.full_like(PG, np.nan)
e_all, t_all = evalp.experiment_id.to_numpy().astype(str), evalp.relative_time_s.to_numpy(float)
folds = [f for f in cv.make_folds(pool, n_repeats=1)]
params = {}
for f in folds:
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
    va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    PL[va] = M.build_model(None).fit(XL_fit[tr], y[tr]).predict_proba(XL_eval[va])
    trp = pool[tr]
    params[f['fold']] = (va, fit_params(trp.label.to_numpy(), trp.relative_time_s.to_numpy(), trp.experiment_id.to_numpy()))
    print('LR fold', f['fold'], '완료', flush=True)
pd.to_pickle({'index': evalp.index.to_numpy(), 'proba': PL}, CACHE / 'seq' / 'proba_lr_r0.pkl')


def oof_of(P, hmm):
    pred = np.zeros(len(P), dtype=int)
    for va, prm in params.values():
        pred[va] = decode(P[va], t_all[va], e_all[va], prm, method='hmm', alpha=1.0) if hmm else P[va].argmax(1) + 1
    return pd.DataFrame({'pred_r0': pred}, index=evalp.index)


ref = oof_of(PG, False)
out = {}
for w in (0.0, 0.2, 0.35, 0.5, 0.7, 1.0):
    for mode in ('arith', 'geo'):
        if w in (0.0, 1.0) and mode == 'geo':
            continue
        P = (1 - w) * PG + w * PL if mode == 'arith' else np.exp((1 - w) * np.log(PG + 1e-9) + w * np.log(PL + 1e-9))
        P = P / P.sum(1, keepdims=True)
        for hmm in (False, True):
            name = f"wLR{w:.2f}_{mode}{'_hmm' if hmm else ''}"
            o = oof_of(P, hmm)
            r = cv.evaluate(evalp, o, n_boot=300)
            c = cv.compare(evalp, o, ref, n_boot=2000)
            out[name] = o
            print(f"{name:22s} CV {r['cv']:.4f} | long {r['long_exps']:.3f} short {r['short_exps']:.3f} | train-origin {r['origin_train']:.3f} val-origin {r['origin_val']:.3f}"
                  f" | per-class {np.round(r['per_class'], 3).tolist()} | vs GBDT {c['diff']:+.4f} p {c['p_a_better']:.2f}", flush=True)

if args.submit:
    test = pd.read_csv(args.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    paths = M.resolve_image_paths(test, [args.data_dir / 'test'])
    gt = M.extract_image_features(test, paths, n_jobs=4)
    pl = M.build_model(None).fit(XL_fit, y).predict_proba(M.build_features(test, gt))
    color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
    g = lgb.LGBMClassifier(learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=50, reg_lambda=1.0,
                           colsample_bytree=0.8, n_jobs=4, random_state=42, verbose=-1).fit(make_X(pool, color, ['color', 'extra']), y)
    pg = g.predict_proba(make_X(test, pd.read_pickle(CACHE / 'test_color.pkl'), ['color', 'extra']))
    pd.to_pickle({'sample_id': test.sample_id.to_numpy(), 'lr': pl, 'gbdt': pg}, CACHE / 'seq' / 'test_proba_lr_gbdt.pkl')
    sub = Path(__file__).resolve().parent.parent.parent / 'submissions'
    prm = fit_params(pool.label.to_numpy(), pool.relative_time_s.to_numpy(), pool.experiment_id.to_numpy())
    te, tt = test.experiment_id.to_numpy().astype(str), test.relative_time_s.to_numpy(float)
    for name, P, hmm in [('sub04_lr_refit97', pl, False), ('sub05_blend_lr50_gbdt50', 0.5 * pl + 0.5 * pg, False),
                         ('sub06_blend_lr50_gbdt50_hmm', 0.5 * pl + 0.5 * pg, True), ('sub07_lgbm_ref_hmm', pg, True)]:
        pred = decode(P, tt, te, prm, method='hmm', alpha=1.0) if hmm else P.argmax(1) + 1
        pd.DataFrame({'sample_id': test.sample_id, 'label': pred.astype(int)}).to_csv(sub / f'{name}.csv', index=False)
        print('저장', name, pd.Series(pred).value_counts(normalize=True).sort_index().round(3).to_dict())
