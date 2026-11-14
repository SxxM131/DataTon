"""무릎점 피처(knee.py)를 최종 후보 모델(tab nocum + 규칙 피처)에 더했을 때의 효과. 공통 검증 5-fold × 1회."""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'common'), str(HERE.parent / 'tab'), str(HERE.parent / 'rules')]
import cv, feats, knee, rules  # noqa: E402

DATA = Path.home() / 'data' / 'battery_data_light'
CACHE = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(DATA)
evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])


def knee_feats(fr):
    def fn(f):
        t = f.relative_time_s.to_numpy(float)
        t1cm = np.maximum.accumulate(f.meas_t1.to_numpy(float))
        gate = t1cm >= 50
        t50 = t[gate][0] if gate.any() else np.inf
        out = {}
        for c in ['meas_p', 'meas_t1', 'meas_t2']:
            x = f[c].to_numpy(float)
            for ref in (90.0, 180.0):
                z, r, s = knee.knee_z(t, x, ref_s=ref)
                k = f'k_{c[5:]}_{int(ref)}'
                out[k + '_z'] = z
                zg = np.where(gate, z, 0.0)
                out[k + '_zcm50'] = np.maximum.accumulate(zg)                     # t1>=50 이후 z 누적 최댓값
                for th in (4, 8):
                    first = np.where(zg >= th, t, np.inf)
                    first = np.minimum.accumulate(first)
                    out[k + f'_since{th}'] = np.where(np.isfinite(first), t - first, -1.0)
            if c == 'meas_p':
                out['k_p_sigma'] = s
                # t1>=50 시점의 압력 대비 변화 (잡음 단위)
                p50 = x[gate][0] if gate.any() else np.nan
                out['k_p_from50_z'] = np.where(gate, (x - p50) / s, 0.0)
        out['k_since50'] = np.where(gate, t - t50, -1.0)
        return out
    return knee.per_experiment(fr, fn)


def F(fr, with_knee):
    X = pd.concat([feats.build(fr, color, {'extra': True, 'cum': False}), rules.rule_features(fr)], axis=1)
    return pd.concat([X, knee_feats(fr)], axis=1) if with_knee else X


P = dict(objective='multiclass', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.3, max_bin=63, reg_lambda=1, n_jobs=6, verbose=-1, random_state=42)
y = pool['label'].to_numpy(int)
res = {}
for name, wk in [('base', False), ('+knee', True)]:
    Xp, Xe = F(pool, wk), F(evalp, wk)
    oof = pd.DataFrame(index=evalp.index)
    for f in cv.make_folds(pool, n_repeats=1):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
        va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        m = lgb.LGBMClassifier(**P).fit(Xp[tr], y[tr])
        oof.loc[evalp.index[va], 'pred_r0'] = m.predict(Xe[va])
    res[name] = oof.astype(int)
    r = cv.evaluate(evalp, res[name], n_boot=300)
    line = f"{name:6s} 피처 {Xp.shape[1]} | CV {r['cv']:.4f} | long {r['long_exps']:.3f} short {r['short_exps']:.3f} | per-class {np.round(r['per_class'], 3).tolist()}"
    if name != 'base':
        c = cv.compare(evalp, res[name], res['base'], n_boot=2000)
        line += f" | vs base {c['diff']:+.4f} [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p {c['p_a_better']:.2f}"
        imp = pd.Series(m.booster_.feature_importance('gain'), index=Xp.columns)
        print('  무릎 피처 중요도 순위 상위:', [(k, int((imp.rank(ascending=False))[k])) for k in imp[imp.index.str.startswith('k_')].sort_values(ascending=False).index[:6]])
    print(line, flush=True)
