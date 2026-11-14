"""2·3단계 재판정 모델을 배터리 종류(원통형/파우치/각형)별로 따로 학습하면 나아지는가. 공통 검증 3회.

기준: 전체 실험으로 학습한 재판정 모델 (st3_presrise_mask_r3 OOF). 비교: 같은 종류 실험만으로 학습한 모델, 두 확률의 평균.
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

C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int); ye = evalp.label.to_numpy(int)
R = HERE.parent / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
PS = np.zeros((3, len(evalp), 6))
PR = {**M.LGB_PARAMS, 'n_jobs': 6}
bc_p, bc_e = pool.bat_class.to_numpy(), evalp.bat_class.to_numpy()
for f in cv.make_folds(pool, n_repeats=3):
    r = f['rep']; tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    Qtr = M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er)); Qva = M.pressure_features(evalp[va], tab)
    Xtr = pd.concat([Xp[tr], Qtr], axis=1); Xva = pd.concat([Xe[va], Qva], axis=1)
    for c in (1, 2, 3):
        mt = bc_p[tr] == c; mv = bc_e[va] == c
        if mv.any():
            m = lgb.LGBMClassifier(**PR).fit(Xtr[mt], y[tr][mt])
            pr = m.predict_proba(Xva[mv]); full = np.zeros((mv.sum(), 6)); full[:, m.classes_ - 1] = pr
            idx = np.flatnonzero(va)[mv]; PS[r, idx] = full
    print('fold', r, f['fold'], flush=True)
np.save(C / 'st3_class_specific.npy', PS)
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
m23 = np.isin(ye, (2, 3))
V = {}
for name in ('전체 학습(현재)', '종류별 학습', '둘의 평균'):
    out = pd.DataFrame(index=evalp.index)
    for r in range(3):
        pb = B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy()
        q = pb if name == '전체 학습(현재)' else PS[r] if name == '종류별 학습' else (pb + PS[r]) / 2
        out[f'pred_r{r}'] = M.apply_rules(A[f'pred_r{r}'].to_numpy(int), q, evalp)
    V[name] = out
base = V['전체 학습(현재)']
rows = []
for c in sorted(set(cfg)):
    m = (cfg == c) & m23
    rows.append({'cfg': c, **{k: round(np.mean([((v[f'pred_r{r}'].to_numpy() != ye) & m).sum() for r in range(3)])) for k, v in V.items()}})
print(pd.DataFrame(rows).to_string(index=False))
for k, v in V.items():
    rr = cv.evaluate(evalp, v, n_boot=100); s = f"{k:14s} CV {rr['cv']:.4f} {np.round(rr['per_class'], 3).tolist()}"
    if k != '전체 학습(현재)':
        c = cv.compare(evalp, v, base, n_boot=3000); s += f" | 현재 대비 {c['diff']:+.4f} [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p {c['p_a_better']:.2f}"
    print(s, flush=True)
