"""분석 전용 (제출 금지): 미래 관측을 쓰면 점수가 얼마나 오르는지 = 규정 위반 모델의 상한 추정.

리더보드 상위 점수(0.94)가 인과적(과거만 사용) 모델로 도달 가능한 수준인지 가늠하기 위한 실험이다.
여기서 만든 피처·모델은 대회 규정(미래 관측·실험 전체 통계 사용 금지)을 위반하므로 최종 모델에 절대 쓰지 않는다.

  causal   : tab 추천 피처 (과거만)
  +future  : 같은 피처를 시간을 뒤집어 한 번 더 계산 (= 미래 방향 윈도우·변화량)
  +global  : 실험 전체 t1+t2 최고점까지 남은 시간, 실험 전체 길이 대비 위치
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'tab')]
import cv  # noqa: E402
import feats  # noqa: E402

DATA = Path.home() / 'data' / 'battery_data_light'
CACHE = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(DATA)
evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
CFG = {'extra': True, 'cum': False}


def build(fr, mode):
    X = feats.build(fr, color, CFG)
    if mode == 'causal':
        return X
    rev = fr.copy()
    rev['relative_time_s'] = -rev['relative_time_s']
    Xf = feats.build(rev, color, CFG).add_suffix('_fut')
    Xf = Xf[[c for c in Xf.columns if any(k in c for k in ('s_fut', 'since', 'rate', 'd'))]]
    X = pd.concat([X, Xf], axis=1)
    if mode == 'future+global':
        g = fr.groupby('experiment_id')
        s = fr['meas_t1'] + fr['meas_t2']
        tpk = fr.loc[s.groupby(fr['experiment_id']).idxmax(), ['experiment_id', 'relative_time_s']].set_index('experiment_id')['relative_time_s']
        X['g_to_peak'] = (fr['relative_time_s'] - fr['experiment_id'].map(tpk)).to_numpy()
        X['g_frac'] = (fr['relative_time_s'] / g['relative_time_s'].transform('max')).to_numpy()
    return X


P = dict(objective='multiclass', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.3, max_bin=63, reg_lambda=1, n_jobs=6, verbose=-1, random_state=42)
y = pool['label'].to_numpy(int)
base = None
for mode in ('causal', 'future', 'future+global'):
    Xp, Xe = build(pool, mode), build(evalp, mode)
    oof = pd.DataFrame(index=evalp.index)
    for f in cv.make_folds(pool, n_repeats=1):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
        va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        oof.loc[evalp.index[va], 'pred_r0'] = lgb.LGBMClassifier(**P).fit(Xp[tr], y[tr]).predict(Xe[va])
    oof = oof.astype(int)
    r = cv.evaluate(evalp, oof, n_boot=300)
    line = f"{mode:14s} 피처 {Xp.shape[1]:4d} | CV {r['cv']:.4f} [{r['boot_ci90'][0]:.3f},{r['boot_ci90'][1]:.3f}] | long {r['long_exps']:.3f} short {r['short_exps']:.3f} | per-class {np.round(r['per_class'], 3).tolist()}"
    if base is not None:
        c = cv.compare(evalp, oof, base, n_boot=1000)
        line += f" | vs causal {c['diff']:+.4f} p {c['p_a_better']:.2f}"
    else:
        base = oof
    print(line, flush=True)
