"""2·3단계 전용 모델(이진 분류) 실험.

기본 6단계 모델(rules_tab_nocum_r3의 OOF 예측)이 2 또는 3이라고 한 행에 대해서만, 2·3단계 행으로 학습한 이진 모델이 다시 판정한다.
피처: 기본 296개 + '압력 연속 상승' 근사 피처(presrise.py, 임계 규칙이 다른 구성은 비움).
변형: --aug  긴 학습 실험의 Test식 다운샘플 복사본을 학습에 추가 (10/15초 간격 대응)
"""
import argparse, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'common'), str(HERE.parent / 'tab'), str(HERE.parent / 'rules')]
import cv, feats, rules, presrise  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--n_repeats', type=int, default=3); ap.add_argument('--aug', action='store_true'); ap.add_argument('--save', default='')
ap.add_argument('--base', default='rules_tab_nocum_r3')
args = ap.parse_args()
DATA = Path.home() / 'data' / 'battery_data_light'; CACHE = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(DATA); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
bp = pd.concat([feats.build(pool, color, {'extra': True, 'cum': False}), rules.rule_features(pool)], axis=1)
be = pd.concat([feats.build(evalp, color, {'extra': True, 'cum': False}), rules.rule_features(evalp)], axis=1)
B = pd.read_pickle(HERE.parent / 'results' / 'oof' / f'{args.base}.pkl').reindex(evalp.index)
P = dict(objective='binary', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.3, max_bin=63, reg_lambda=1, n_jobs=6, verbose=-1, random_state=42, deterministic=True, force_row_wise=True)
y = pool['label'].to_numpy(int); ye = evalp['label'].to_numpy(int)
et = cv.exp_table(pool); long_exps = set(et.index[et.long])
out = B[[c for c in B.columns if c.startswith('pred_r')]].astype(int).copy()
prob = pd.DataFrame(index=evalp.index)
for f in cv.make_folds(pool, n_repeats=args.n_repeats):
    r = f['rep']
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    er = presrise.experiment_ranges(pool[tr]); tab = presrise.range_table(er); loo = presrise.loo_ranges(er)
    Xtr = pd.concat([bp[tr], presrise.features(pool[tr], tab, own_range=loo, mask_special=True)], axis=1)
    ytr = y[tr]
    if args.aug:                                    # 긴 학습 실험의 Test식 행 (피처는 그 행들만 보고 생성)
        tl = evalp.experiment_id.isin(f['train_exps']).to_numpy() & evalp.experiment_id.isin(long_exps).to_numpy()
        Xa = pd.concat([be[tl], presrise.features(evalp[tl], tab, own_range=loo, mask_special=True)], axis=1)
        Xtr = pd.concat([Xtr, Xa]); ytr = np.concatenate([ytr, ye[tl]])
    m23 = (ytr == 2) | (ytr == 3)
    m = lgb.LGBMClassifier(**P).fit(Xtr[m23], (ytr[m23] == 3).astype(int))
    Xva = pd.concat([be[va], presrise.features(evalp[va], tab, mask_special=True)], axis=1)
    p3 = m.predict_proba(Xva)[:, 1]
    prob.loc[evalp.index[va], f'p3_r{r}'] = p3
    b = out.loc[evalp.index[va], f'pred_r{r}'].to_numpy()
    sel = (b == 2) | (b == 3)
    c = b.copy(); c[sel] = np.where(p3[sel] >= 0.5, 3, 2)
    out.loc[evalp.index[va], f'pred_r{r}'] = c
base = B[[c for c in out.columns]].astype(int)
r0 = cv.evaluate(evalp, base, n_boot=300); r1 = cv.evaluate(evalp, out, n_boot=500); c = cv.compare(evalp, out, base, n_boot=4000)
print(f"기존                 CV {r0['cv']:.4f} | per-class {np.round(r0['per_class'], 3).tolist()}")
print(f"2·3 전용 이진 모델{'(+aug)' if args.aug else ''} CV {r1['cv']:.4f} [{r1['boot_ci90'][0]:.3f},{r1['boot_ci90'][1]:.3f}] | long {r1['long_exps']:.3f} short {r1['short_exps']:.3f} | per-class {np.round(r1['per_class'], 3).tolist()}"
      f" | vs 기존 {c['diff']:+.4f} [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p {c['p_a_better']:.2f}")
if args.save:
    cv.save_result(args.save, r1, pd.concat([out, prob], axis=1))
