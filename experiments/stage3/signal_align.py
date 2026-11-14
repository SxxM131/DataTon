"""3단계 시작과 시점이 맞는 신호 찾기: 각 신호가 '1단계 동안의 자기 기준선'에서 벗어나는 첫 시각 − 3단계 시작 시각.

기준선·잡음은 표면 온도가 45℃에 닿기 전(1단계) 관측에서만 구한다 → 과거 정보만.
이탈: 기준선 중앙값에서 k × (1단계 잡음) 넘게 벗어난 상태가 2행 연속. 3단계 시작과 ±15초 이내면 '맞음'.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light')
SIG = ['meas_p', 'meas_t2', 'thermal_t', 'meas_v', 'indu_t', 'atmp_t', 'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
rows = []
for e, g in pool.groupby('experiment_id', sort=True):
    g = g.sort_values('relative_time_s'); t = g.relative_time_s.to_numpy(float); y = g.label.to_numpy(int); t1 = g.meas_t1.to_numpy(float)
    if not ((y == 3).any() and (y == 2).any()):
        continue
    i2 = int(np.argmax(y == 2)); i3 = int(np.argmax(y == 3))
    cfg = f"{int(g.bat_class.iloc[0])}/{g.bat_cc.iloc[0]:g}/{int(g.tr_method.iloc[0])}"
    for s in SIG:
        x = pd.to_numeric(g[s], errors='coerce').to_numpy(float)
        b = x[:i2]
        if len(b) < 10 or np.isnan(b).mean() > 0.5:
            continue
        med = np.nanmedian(b); d = np.diff(b[~np.isnan(b)])
        sd = max(1.4826 * np.nanmedian(np.abs(b - med)), np.nanstd(d) if len(d) > 2 else 0.0, 1e-6)
        r = {'cfg': cfg, 'sig': s, '2단계(s)': t[i3] - t[i2]}
        for k in (3, 5, 10):
            dev = np.abs(x - med) > k * sd
            hit = dev[1:] & dev[:-1]; hit = np.r_[False, hit]; hit[:i2] = False          # 2단계 이후에서만 찾는다
            r[f'k{k}'] = (t[int(np.argmax(hit))] - t[i3]) if hit.any() else np.nan
        rows.append(r)
d = pd.DataFrame(rows)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
def summ(x):
    x = x.dropna()
    return pd.Series({'n': len(x), '±15초': (x.abs() <= 15).mean() if len(x) else np.nan, '±30초': (x.abs() <= 30).mean() if len(x) else np.nan, '중앙': x.median() if len(x) else np.nan, 'IQR': (x.quantile(.75) - x.quantile(.25)) if len(x) else np.nan})
print('== 전체 97개 실험: 신호별 이탈 시각 − 3단계 시작 (초). ±15초 비율이 높고 IQR이 작을수록 좋음 ==')
out = []
for s in SIG:
    for k in (3, 5, 10):
        r = summ(d[d.sig == s][f'k{k}']); r['sig'] = s; r['k'] = k; out.append(r)
o = pd.DataFrame(out).sort_values('±15초', ascending=False)
print(o[['sig', 'k', 'n', '±15초', '±30초', '중앙', 'IQR']].head(14).round(2).to_string(index=False))
print('\n참고: 2단계 길이의 중앙값', d.drop_duplicates(['cfg', '2단계(s)'])['2단계(s)'].median(), '초 (아무 정보 없이 2단계 시작만 알면 ±15초 안에 들 확률은 낮다)')
print('\n== 구성별 최고 신호 (±15초 비율, n ≥ 실험 수의 60%) ==')
for c, gc in d.groupby('cfg'):
    best = []
    nexp = gc.groupby('sig').size().max()
    for s in SIG:
        for k in (3, 5, 10):
            x = gc[gc.sig == s][f'k{k}'].dropna()
            if len(x) >= 0.6 * nexp:
                best.append((float((x.abs() <= 15).mean()), float((x.abs() <= 30).mean()), s, k, len(x), float(x.median())))
    best.sort(reverse=True)
    print(f"  {c:9s} (실험 {nexp:2d}): " + ' | '.join(f"{s} k{k}: ±15초 {a:.0%}, ±30초 {b:.0%}, 중앙 {m:+.0f}s (n={n})" for a, b, s, k, n, m in best[:3]))
