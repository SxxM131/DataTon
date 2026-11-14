"""부호 기반 압력 피처(signfeat.py)를 최종 후보 모델에 더했을 때의 효과. 공통 검증."""
import argparse, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'common'), str(HERE.parent / 'tab'), str(HERE.parent / 'rules')]
import cv, feats, rules, signfeat  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--n_repeats', type=int, default=1); ap.add_argument('--save', default='')
args = ap.parse_args()
DATA = Path.home() / 'data' / 'battery_data_light'; CACHE = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(DATA); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])


def F(fr, ws):
    X = pd.concat([feats.build(fr, color, {'extra': True, 'cum': False}), rules.rule_features(fr)], axis=1)
    return pd.concat([X, signfeat.sign_features(fr)], axis=1) if ws else X


P = dict(objective='multiclass', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.3, max_bin=63, reg_lambda=1, n_jobs=6, verbose=-1, random_state=42,
         deterministic=True, force_row_wise=True)
y = pool['label'].to_numpy(int)
res = {}
for name, ws in [('base', False), ('+sign', True)]:
    Xp, Xe = F(pool, ws), F(evalp, ws)
    oof = pd.DataFrame(index=evalp.index)
    for f in cv.make_folds(pool, n_repeats=args.n_repeats):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        m = lgb.LGBMClassifier(**P).fit(Xp[tr], y[tr])
        pr = m.predict_proba(Xe[va]); r = f['rep']
        oof.loc[evalp.index[va], f'pred_r{r}'] = pr.argmax(1) + 1
        for k in range(6):
            oof.loc[evalp.index[va], f'p{k+1}_r{r}'] = pr[:, k]
    for c in oof.columns:
        if c.startswith('pred'):
            oof[c] = oof[c].astype(int)
    res[name] = oof
    r = cv.evaluate(evalp, oof, n_boot=300)
    line = f"{name:6s} 피처 {Xp.shape[1]} | CV {r['cv']:.4f} [{r['boot_ci90'][0]:.3f},{r['boot_ci90'][1]:.3f}] | long {r['long_exps']:.3f} short {r['short_exps']:.3f} | per-class {np.round(r['per_class'], 3).tolist()}"
    if name != 'base':
        c = cv.compare(evalp, oof, res['base'], n_boot=2000)
        line += f" | vs base {c['diff']:+.4f} [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p {c['p_a_better']:.2f}"
        imp = pd.Series(m.booster_.feature_importance('gain'), index=Xp.columns); rk = imp.rank(ascending=False)
        print('  부호 피처 중요도 순위 상위:', [(k, int(rk[k])) for k in imp[imp.index.str.startswith('s_')].sort_values(ascending=False).index[:8]])
        if args.save:
            cv.save_result(args.save, r, oof)
    print(line, flush=True)
