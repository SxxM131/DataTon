"""실험 묶음 배깅: 학습 실험의 80%를 무작위로 뽑아 학습한 모델 K개의 확률 평균. 공통 검증(반복 0~1).

시드 평균(행·피처 샘플링만 다름)과 달리 '어떤 실험으로 학습했는가'에 따른 흔들림을 줄이는 것이 목적 (Private 방어).
"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

K = 5
C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int); pe = pool.experiment_id.to_numpy()
PA = np.zeros((2, K, len(evalp), 6), np.float32); PB = np.zeros_like(PA)
rng = np.random.default_rng(7)
for f in cv.make_folds(pool, n_repeats=2):
    r = f['rep']; va = evalp.experiment_id.isin(f['valid_exps']).to_numpy(); tr_exps = np.array(sorted(f['train_exps']))
    for k in range(K):
        sub = rng.choice(tr_exps, int(round(0.8 * len(tr_exps))), replace=False); tr = np.isin(pe, sub)
        er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
        P = {**M.LGB_PARAMS, 'n_jobs': 6, 'random_state': 42 + k}
        PA[r, k, va] = lgb.LGBMClassifier(**P).fit(Xp[tr], y[tr]).predict_proba(Xe[va])
        Xb = pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1)
        PB[r, k, va] = lgb.LGBMClassifier(**P).fit(Xb, y[tr]).predict_proba(pd.concat([Xe[va], M.pressure_features(evalp[va], tab)], axis=1))
    print('fold', r, f['fold'], flush=True)
    np.savez_compressed(C / 'robust_exp_bag.npz', PA=PA, PB=PB, index=evalp.index.to_numpy())
print('done', flush=True)
