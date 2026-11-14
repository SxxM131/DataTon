"""'압력 연속 상승' 근사 피처(presrise.py)를 최종 후보 모델에 더했을 때의 효과. 공통 검증."""
import argparse, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'common'), str(HERE.parent / 'tab'), str(HERE.parent / 'rules')]
import cv, feats, rules, presrise  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--n_repeats', type=int, default=1); ap.add_argument('--save', default=''); ap.add_argument('--mask', action='store_true')
args = ap.parse_args()
DATA = Path.home() / 'data' / 'battery_data_light'; CACHE = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(DATA); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
base_p = pd.concat([feats.build(pool, color, {'extra': True, 'cum': False}), rules.rule_features(pool)], axis=1)
base_e = pd.concat([feats.build(evalp, color, {'extra': True, 'cum': False}), rules.rule_features(evalp)], axis=1)
P = dict(objective='multiclass', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.3, max_bin=63, reg_lambda=1, n_jobs=6, verbose=-1, random_state=42,
         deterministic=True, force_row_wise=True)
y = pool['label'].to_numpy(int)
oofs = {'base': pd.DataFrame(index=evalp.index), '+presrise': pd.DataFrame(index=evalp.index)}
for f in cv.make_folds(pool, n_repeats=args.n_repeats):
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    er = presrise.experiment_ranges(pool[tr])                       # 학습 fold 실험에서만 범위 통계 산출
    tab = presrise.range_table(er)
    q_tr = presrise.features(pool[tr], tab, own_range=presrise.loo_ranges(er), mask_special=args.mask)
    q_va = presrise.features(evalp[va], tab, mask_special=args.mask)
    for name in oofs:
        Xtr = base_p[tr] if name == 'base' else pd.concat([base_p[tr], q_tr], axis=1)
        Xva = base_e[va] if name == 'base' else pd.concat([base_e[va], q_va], axis=1)
        m = lgb.LGBMClassifier(**P).fit(Xtr, y[tr])
        pr = m.predict_proba(Xva); r = f['rep']
        oofs[name].loc[evalp.index[va], f'pred_r{r}'] = pr.argmax(1) + 1
        for k in range(6):
            oofs[name].loc[evalp.index[va], f'p{k+1}_r{r}'] = pr[:, k]
    print('fold', f['rep'], f['fold'], flush=True)
for name, o in oofs.items():
    for c in o.columns:
        if c.startswith('pred'):
            o[c] = o[c].astype(int)
    r = cv.evaluate(evalp, o, n_boot=300)
    line = f"{name:10s} CV {r['cv']:.4f} [{r['boot_ci90'][0]:.3f},{r['boot_ci90'][1]:.3f}] | long {r['long_exps']:.3f} short {r['short_exps']:.3f} | bat1/2/3 {r['bat1']:.3f}/{r['bat2']:.3f}/{r['bat3']:.3f} | per-class {np.round(r['per_class'], 3).tolist()}"
    if name != 'base':
        c = cv.compare(evalp, o, oofs['base'], n_boot=2000)
        line += f" | vs base {c['diff']:+.4f} [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p {c['p_a_better']:.2f}"
        imp = pd.Series(m.booster_.feature_importance('gain'), index=Xtr.columns); rk = imp.rank(ascending=False)
        print('  새 피처 중요도 순위 상위:', [(k, int(rk[k])) for k in imp[imp.index.str.startswith('q_')].sort_values(ascending=False).index[:8]])
        if args.save:
            cv.save_result(args.save, r, o)
    print(line, flush=True)
