"""원형 4.8Ah 가열(Test에 4개)의 2→3단계 전환 오차 구조. OOF 예측의 '첫 3단계 시각' − 실제 시각(초)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
R = HERE.parent / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index)
B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
sel = (evalp.bat_class == 1) & (evalp.bat_cc == 4.8)
rows = []
for k, (e, g) in enumerate(evalp[sel].groupby('experiment_id', sort=True)):
    g = g.sort_values('relative_time_s'); t = g.relative_time_s.to_numpy(float); y = g.label.to_numpy(int)
    if not (y == 3).any():
        continue
    t3 = t[y == 3][0]; t2 = t[y == 2][0] if (y == 2).any() else np.nan
    full = pool[pool.experiment_id == e].sort_values('relative_time_s')
    p = full.meas_p.to_numpy(float); tf = full.relative_time_s.to_numpy(float); t1 = full.meas_t1.to_numpy(float)
    rng = p.max() - p.min(); i3 = np.searchsorted(tf, t3)
    r = {'#': k, 'n': len(g), 'dt': float(np.median(np.diff(t))), 'soc': int(g.soc_status.iloc[0]), '2단계 길이(s)': t3 - t2, 't1@3': round(t1[i3], 1),
         'p0': round(p[0], 1), '범위/p0': round(rng / max(abs(p[0]), 1e-3), 2), 'p끝/p0': round(p[-1] / p[0], 2)}
    for nm, O in (('A', A), ('B', B)):
        errs = []
        for rep in range(3):
            pr = O.loc[g.index, f'pred_r{rep}'].to_numpy(int)
            if nm == 'B':                       # 재판정: A가 2/3인 행만 B의 p2 vs p3
                a = A.loc[g.index, f'pred_r{rep}'].to_numpy(int)
                b = np.where(O.loc[g.index, f'p3_r{rep}'].to_numpy() > O.loc[g.index, f'p2_r{rep}'].to_numpy(), 3, 2)
                pr = np.where((a == 2) | (a == 3), b, a)
            f3 = t[pr >= 3][0] if (pr >= 3).any() else np.nan
            errs.append(f3 - t3)
        r[f'{nm} 오차(s)'] = '/'.join(f'{x:+.0f}' for x in errs)
        r[f'{nm} 오류행'] = round(np.mean([((O.loc[g.index, f'pred_r{rep}'].to_numpy(int) != y) & (y <= 3)).sum() for rep in range(3)]), 1) if nm == 'A' else ''
    rows.append(r)
d = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
print(d.to_string(index=False))
