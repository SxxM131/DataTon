"""과거만 쓰는 '압력 상승 규칙'으로 3단계 시작을 직접 정했을 때의 2·3단계 오류 행 수 (구성별), 모델과 비교.

규칙: 온도 50~90℃ 구간에서 p(t) − p(t−w) ≥ c × R̂ 가 처음 참이 된 시각부터 3단계.
R̂ (모두 '자기 실험을 뺀' 같은 구성 학습 실험에서 산출): med = 범위 중앙값, q25 = 범위 25% 분위, p0k = (범위/초기압력 중앙값) × 초기압력,
  p0k25 = (범위/초기압력 25% 분위) × 초기압력
평가 행: Test식 행(evalp) 중 실제 라벨이 2 또는 3인 행. 모델 쪽은 3회 반복 평균.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
R_ = HERE.parent / 'results' / 'oof'
A = pd.read_pickle(R_ / 'rules_tab_nocum_r3.pkl').reindex(evalp.index)
B = pd.read_pickle(R_ / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
KEY = ['bat_class', 'bat_cc', 'tr_method']
g = pool.groupby('experiment_id')
meta = g[KEY + ['soc_status']].first()
meta['R'] = g.meas_p.max() - g.meas_p.min()
meta['p0'] = pool.sort_values('relative_time_s').groupby('experiment_id').meas_p.first().abs().clip(lower=1e-3)
meta['k'] = meta.R / meta.p0
meta['cfg'] = meta[KEY].astype(float).astype(str).agg('/'.join, axis=1)
special = ((meta.bat_class == 3) & (meta.bat_cc == 280) & (meta.tr_method == 1)) | ((meta.bat_class == 2) & (meta.bat_cc == 73) & (meta.tr_method == 1) & (meta.soc_status == 1))

RULES = [(est, c, w) for est in ('med', 'q25', 'p0k', 'p0k25') for c in (0.01, 0.014, 0.02) for w in (20,)] + [('med', 0.014, 10), ('q25', 0.02, 10), ('med', 0.02, 30)]
rows = []
for e, ge in evalp.groupby('experiment_id', sort=True):
    m = meta.loc[e]
    ge = ge.sort_values('relative_time_s')
    t = ge.relative_time_s.to_numpy(float); y = ge.label.to_numpy(int); p = ge.meas_p.to_numpy(float); t1 = ge.meas_t1.to_numpy(float)
    in23 = (y == 2) | (y == 3)
    if not in23.any():
        continue
    oth = meta[(meta.cfg == m.cfg) & (meta.index != e)]
    est = {'med': oth.R.median(), 'q25': oth.R.quantile(.25), 'p0k': oth.k.median() * m.p0, 'p0k25': oth.k.quantile(.25) * m.p0}
    r = {'cfg': m.cfg, 'special': bool(special[e]), 'n23': int(in23.sum())}
    for nm, O in (('A', A), ('A+B', B)):
        errs = []
        for rep in range(3):
            pr = A.loc[ge.index, f'pred_r{rep}'].to_numpy(int)
            if nm == 'A+B':
                b = np.where(O.loc[ge.index, f'p3_r{rep}'].to_numpy() > O.loc[ge.index, f'p2_r{rep}'].to_numpy(), 3, 2)
                pr = np.where((pr == 2) | (pr == 3), b, pr)
            errs.append(((pr != y) & in23).sum())
        r[nm] = float(np.mean(errs))
    for (en, c, w) in RULES:
        j = np.searchsorted(t, t - w, side='right') - 1; ok = j >= 0
        d = np.zeros(len(t)); d[ok] = p[ok] - p[j[ok]]
        t1b = np.full(len(t), np.nan); t1b[ok] = t1[j[ok]]
        fire = ok & (t1b >= 50) & (t1b <= 90) & (d >= c * est[en] * (w / 20 if w < 20 else 1))
        fired = np.maximum.accumulate(fire)
        # 4단계 이후는 다른 규칙 몫: 여기서는 2·3 행만 본다. 규칙이 아직 안 켜졌으면 2, 켜졌으면 3
        pr = np.where(fired, 3, 2)
        r[f'{en}|{c}|{w}'] = float(((pr != y) & in23).sum())
    rows.append(r)
d = pd.DataFrame(rows)
cols = [c for c in d.columns if c not in ('cfg', 'special', 'n23')]
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
tab = d.groupby(['cfg', 'special']).agg(n_exp=('n23', 'size'), n23=('n23', 'sum'), **{c: (c, 'sum') for c in cols}).round(0).astype(int)
print(tab.T.to_string())
print('\n비특수 구성 합계:', d[~d.special][cols].sum().round(0).astype(int).sort_values().to_dict())
d.to_pickle(Path.home() / 'jaehyeok' / 'cache' / 'rule_onset_per_exp.pkl')
