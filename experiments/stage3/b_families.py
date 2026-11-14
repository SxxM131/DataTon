"""2·3단계 재판정 모델을 다른 계열 모델과 섞으면 나아지는가. 공통 검증 3회, 최종 규칙 체인 적용.

현재: LightGBM 6단계 모델의 p2·p3. 후보:
  xgb   — XGBoost 6단계 (깊이 기준 트리)
  et    — ExtraTrees 6단계 (무작위 분할, 결측은 −999로 채움)
  mono  — LightGBM 이진(3 vs 2), 누적형 피처(경과 시간·누적 최대 상승량 등)에 단조 증가 제약
각각 단독, 그리고 현재 모델과 확률 평균.
"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier

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
Q3 = {k: np.full((3, len(evalp)), np.nan) for k in ('xgb', 'et', 'mono')}          # p3/(p2+p3)
for f in cv.make_folds(pool, n_repeats=3):
    r = f['rep']; tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    Xtr = pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1)
    Xva = pd.concat([Xe[va], M.pressure_features(evalp[va], tab)], axis=1)
    ytr = y[tr]
    m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.3, min_child_weight=5, tree_method='hist', max_bin=64,
                          n_jobs=6, random_state=42, verbosity=0).fit(Xtr, ytr - 1)
    p = m.predict_proba(Xva); Q3['xgb'][r, va] = p[:, 2] / (p[:, 1] + p[:, 2] + 1e-12)
    m = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, max_features=0.3, n_jobs=6, random_state=42).fit(Xtr.fillna(-999.0), ytr)
    p = m.predict_proba(Xva.fillna(-999.0)); Q3['et'][r, va] = p[:, 2] / (p[:, 1] + p[:, 2] + 1e-12)
    m23 = (ytr == 2) | (ytr == 3)
    mono_cols = [c for c in Xtr.columns if c.endswith('_cm') or c.endswith('_cmc') or c.endswith('_p0cm') or c.endswith('_p0cmc') or c.startswith('q_since') or c.startswith('q_sincec')
                 or c in ('since_t1_45', 'r_since_t1_45', 'r_since_t1_50', 'r_t1_over50', 'q_since_t50', 'q_d20_cm_doc', 'q_d20_cmc_doc', 'r_since_p_rise')]
    cons = [1 if c in mono_cols else 0 for c in Xtr.columns]
    m = lgb.LGBMClassifier(objective='binary', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
                           max_bin=63, reg_lambda=1.0, n_jobs=6, verbose=-1, deterministic=True, force_row_wise=True, random_state=42,
                           monotone_constraints=cons, monotone_constraints_method='intermediate').fit(Xtr[m23], (ytr[m23] == 3).astype(int))
    Q3['mono'][r, va] = m.predict_proba(Xva)[:, 1]
    print('fold', r, f['fold'], flush=True)
np.savez_compressed(C / 'st3_b_families.npz', **Q3)
qb = np.stack([B[f'p3_r{r}'].to_numpy() / (B[f'p2_r{r}'].to_numpy() + B[f'p3_r{r}'].to_numpy() + 1e-12) for r in range(3)])
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])
m23e = np.isin(ye, (2, 3))


def final(q):
    out = pd.DataFrame(index=evalp.index)
    for r in range(3):
        pb = np.zeros((len(evalp), 6)); pb[:, 1] = 1 - q[r]; pb[:, 2] = q[r]
        out[f'pred_r{r}'] = M.apply_rules(A[f'pred_r{r}'].to_numpy(int), pb, evalp)
    return out


base = final(qb)
V = {'현재 (LightGBM 6단계)': qb}
for k in Q3:
    V[f'{k} 단독'] = Q3[k]; V[f'현재 + {k} 평균'] = (qb + Q3[k]) / 2
V['현재 + xgb + et 평균'] = (qb + Q3['xgb'] + Q3['et']) / 3
V['전부 평균'] = (qb + Q3['xgb'] + Q3['et'] + Q3['mono']) / 4
for k, q in V.items():
    o = final(q); rr = cv.evaluate(evalp, o, n_boot=100); e23 = np.mean([((o[f'pred_r{r}'].to_numpy() != ye) & m23e).sum() for r in range(3)]); e23t = np.mean([((o[f'pred_r{r}'].to_numpy() != ye) & m23e & TEST).sum() for r in range(3)])
    s = f"{k:22s} CV {rr['cv']:.4f} | 2·3 오류 행 {e23:6.0f} (Test 구성 {e23t:5.0f}) | 2·3단계 F1 {rr['per_class'][1]:.3f}/{rr['per_class'][2]:.3f}"
    if k != '현재 (LightGBM 6단계)':
        c = cv.compare(evalp, o, base, n_boot=3000); s += f" | 현재 대비 {c['diff']:+.4f} [{c['ci90'][0]:+.4f},{c['ci90'][1]:+.4f}] p {c['p_a_better']:.2f}"
    print(s, flush=True)
