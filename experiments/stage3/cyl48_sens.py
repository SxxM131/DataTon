"""원형 4.8Ah: 문서 규칙(20초 뒤 압력 상승 ≥ 0.02 × 범위, 온도 50~90℃)이 라벨과 맞는지, 그리고 임계가 틀리면 시작 시각이 얼마나 밀리는지.
분석 전용 (실험 전체 범위·미래 값을 쓴다)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light')
sel = pool[(pool.bat_class == 1) & (pool.bat_cc == 4.8)]
rows = []
for k, (e, g) in enumerate(sel.groupby('experiment_id', sort=True)):
    g = g.sort_values('relative_time_s'); t = g.relative_time_s.to_numpy(float); y = g.label.to_numpy(int)
    p = g.meas_p.to_numpy(float); t1 = g.meas_t1.to_numpy(float)
    if not (y == 3).any():
        continue
    t3 = t[y == 3][0]; R = p.max() - p.min(); p0 = max(abs(p[0]), 1e-3)
    j = np.searchsorted(t, t + 20, side='left'); ok = j < len(t)
    fwd = np.full(len(t), np.nan); fwd[ok] = p[j[ok]] - p[ok]                 # 20초 뒤 압력 − 현재 (미래 값, 분석용)
    gate = (t1 >= 50) & (t1 <= 90)

    def first(thr):
        m = gate & (np.nan_to_num(fwd, nan=-np.inf) >= thr)
        return t[m][0] if m.any() else np.nan

    r = {'#': k, 'p0': round(p0, 1), 'R/p0': round(R / p0, 2), '0.02R': round(0.02 * R, 2), '규칙 시각−라벨(s)': first(0.02 * R) - t3}
    for f in (0.5, 0.7, 1.4, 2.0):
        r[f'임계×{f}'] = first(0.02 * R * f) - t3
    r['p0기준(×1.63)'] = first(0.02 * 1.63 * p0) - t3                          # 범위 = 1.63 × p0 (구성 중앙값)로 가정
    r['구성 중앙값 R'] = first(0.02 * 36.65) - t3
    # 지금까지 관측한 범위(누적)로 정규화: 시각 t에서 알 수 있는 값만
    run = np.maximum.accumulate(p) - np.minimum.accumulate(p)
    r['t3 시점 누적범위/R'] = round(run[np.searchsorted(t, t3)] / R, 2)
    rows.append(r)
d = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(d.to_string(index=False))
print('\n|오차| ≤ 20초인 실험 수 (18개 중):', {c: int((d[c].abs() <= 20).sum()) for c in d.columns if '임계' in c or '규칙' in c or '기준' in c or '중앙값' in c})
print('|오차| 중앙값(초):', {c: float(d[c].abs().median()) for c in d.columns if '임계' in c or '규칙' in c or '기준' in c or '중앙값' in c})
