"""경과 시간 피처 변형 × 견고성: 평상시(공통 검증 rep 0)와 '느려진 단계' 합성 실험(3단계 ×3, 5단계 ×3)을 함께 본다.

변형 (기본 모델·재판정 모델 둘 다 같은 변형으로 학습):
  base   현재 피처
  noT    relative_time_s(실험 시작 후 경과 초) 제거
  capN   noT + 온도·압력 관련 경과 시간 피처(since_t1_*, since_t2_*, r_since_t1_45/50, r_since_p_rise, q_since_*)를 N초에서 자름
원시 예측(규칙 보정 전)을 저장해 규칙 변형을 따로 평가할 수 있게 한다.
"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402
from dilate_test import dilate, pool, evalp, color  # noqa: E402

TEMP_SINCE = lambda cols: [c for c in cols if c.startswith(('since_t1_', 'since_t2_', 'q_since', 'q_sincec', 'q_sincep0')) or c in ('r_since_t1_45', 'r_since_t1_50', 'r_since_p_rise')]  # noqa: E731


def variant(X, name):
    if name == 'base':
        return X
    X = X.drop(columns=['relative_time_s'])
    if name.startswith('cap'):
        n = float(name[3:]); cs = TEMP_SINCE(X.columns); X = X.copy(); X[cs] = X[cs].clip(upper=n)
    return X


VARS = ['base', 'noT', 'cap300', 'cap120']
KEEP = ['experiment_id', 'sample_id', 'relative_time_s', 'meas_t1', 'meas_t2', 'meas_v', 'bat_class', 'bat_cc', 'tr_method', 'soc_status', 'label']
yp = pool.label.to_numpy(int); PR = {**M.LGB_PARAMS, 'n_jobs': 6}; Xp = M.build_features(pool, color); out = []
for f in cv.make_folds(pool, n_repeats=1):
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp[evalp.experiment_id.isin(f['valid_exps'])]
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    Xb = pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1)
    models = {v: (lgb.LGBMClassifier(**PR).fit(variant(Xp[tr], v), yp[tr]), lgb.LGBMClassifier(**PR).fit(variant(Xb, v), yp[tr])) for v in VARS}
    for s, K in ((0, 1), (3, 2), (3, 3), (5, 2), (5, 3)):
        parts = [dilate(g, s if s else 3, K) for _, g in va.groupby('experiment_id', sort=True)] if s else [g.assign(src_id=g.sample_id.astype(str)) for _, g in va.groupby('experiment_id', sort=True)]
        fr = pd.concat([x for x in parts if x is not None], ignore_index=True)
        col = color.reindex(fr.src_id.values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
        X = M.build_features(fr, col); Xq = pd.concat([X, M.pressure_features(fr, tab)], axis=1)
        o = fr[KEEP].copy(); o['s'] = s; o['K'] = K; o['fold'] = f['fold']
        for v, (a, b) in models.items():
            o[f'a_{v}'] = a.predict(variant(X, v)); pb = b.predict_proba(variant(Xq, v)); o[f'b2_{v}'] = pb[:, 1]; o[f'b3_{v}'] = pb[:, 2]
        out.append(o)
    print('fold', f['fold'], flush=True)
d = pd.concat(out, ignore_index=True); d.to_pickle(Path.home() / 'jaehyeok' / 'cache' / 'robust_time_variants.pkl')
from sklearn.metrics import f1_score
rows = []
for (s, K), x in d.groupby(['s', 'K']):
    x = x.reset_index(drop=True)
    for v in VARS:
        pb = np.zeros((len(x), 6)); pb[:, 1] = x[f'b2_{v}']; pb[:, 2] = x[f'b3_{v}']
        p = M.apply_rules(x[f'a_{v}'].to_numpy(int), pb, x); f = f1_score(x.label, p, labels=range(1, 7), average=None, zero_division=0)
        rows.append({'상황': '평상시' if s == 0 else f'{s}단계 ×{K} 느림', '변형': v, 'Macro F1': round(f.mean(), 4), '2·3 F1': f'{f[1]:.3f}/{f[2]:.3f}', '4·5·6 F1': f'{f[3]:.3f}/{f[4]:.3f}/{f[5]:.3f}'})
pd.set_option('display.width', 200); print(pd.DataFrame(rows).to_string(index=False))
