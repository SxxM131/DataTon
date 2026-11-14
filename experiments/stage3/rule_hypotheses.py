"""3단계 규칙의 해석 후보 비교 (학습 라벨 분석, 미래 값 사용 = 분석 전용).

각 후보는 행마다 통계 g(t)를 만들고, 온도 50~90℃에서 g(t) ≥ c 인 첫 행을 3단계 시작으로 본다. c는 후보마다 최적값을 고른다.
점수: 라벨의 3단계 시작 행과 ±1행 이내인 실험 비율 (특수 구성 2종 제외 82개 실험).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light')
E = []
for e, g in pool.groupby('experiment_id', sort=True):
    g = g.sort_values('relative_time_s'); t = g.relative_time_s.to_numpy(float); p = g.meas_p.to_numpy(float); t1 = g.meas_t1.to_numpy(float); y = g.label.to_numpy(int); v = g.meas_v.to_numpy(float)
    if not (y == 3).any():
        continue
    bc, cc, tm, soc = int(g.bat_class.iloc[0]), float(g.bat_cc.iloc[0]), int(g.tr_method.iloc[0]), int(g.soc_status.iloc[0])
    if (bc == 3 and cc == 280 and tm == 1) or (bc == 2 and cc == 73 and tm == 1 and soc == 1):
        continue
    n = len(t); i3 = int(np.argmax(y == 3)); gate = (t1 >= 50) & (t1 <= 90); eps = 1e-6
    win = (t1 >= 50) & (t1 <= 90); i90 = int(np.argmax(t1 > 90)) if (t1 > 90).any() else n
    dv = np.r_[0, np.diff(v)]; ivd = int(np.argmax((dv <= -0.56) | (v < 2.5))) if ((dv <= -0.56) | (v < 2.5)).any() else n
    rng = lambda a: max(float(np.max(a) - np.min(a)), eps) if len(a) else eps  # noqa: E731
    N = {'전체 범위': rng(p), '50~90℃ 구간 범위': rng(p[win]), '90℃ 도달 전까지 범위': rng(p[:max(i90, 2)]), '전압 급락 전까지 범위': rng(p[:max(ivd, 2)]),
         '초기값': max(abs(p[0]), 1e-3), '상수': 1.0}
    G = {}
    for w in (10.0, 20.0, 30.0):
        j = np.searchsorted(t, t + w, side='left'); ok = j < n; F = np.full(n, np.nan); F[ok] = p[j[ok]] - p[ok]
        for k, d in N.items():
            G[(k, w, '차이')] = F / d
        G[('현재값 대비 비율', w, '차이')] = F / np.maximum(np.abs(p), 1e-3)
        # 연속 상승: 창 안의 5초 단계가 모두 오름
        steps = int(w // 5); up = np.ones(n, bool)
        for s in range(steps):
            a = np.searchsorted(t, t + 5 * s, side='left'); b = np.searchsorted(t, t + 5 * (s + 1), side='left'); o2 = (a < n) & (b < n)
            inc = np.zeros(n, bool); inc[o2] = p[b[o2]] > p[a[o2]]; up &= inc
        for k in ('전체 범위', '50~90℃ 구간 범위', '90℃ 도달 전까지 범위', '초기값'):
            G[(k, w, '연속 상승')] = np.where(up, F / N[k], -np.inf)
        G[('현재값 대비 비율', w, '연속 상승')] = np.where(up, F / np.maximum(np.abs(p), 1e-3), -np.inf)
    E.append({'cfg': f'{bc}/{cc:g}/{tm}', 'i3': i3, 'gate': gate, 'G': G})
print('평가 실험', len(E), '개')
grid = np.exp(np.linspace(np.log(1e-4), np.log(20.0), 160))
def score(key, c, sub=E):
    d = []
    for x in sub:
        m = x['gate'] & (np.nan_to_num(x['G'][key], nan=-np.inf) >= c); d.append(int(np.argmax(m)) - x['i3'] if m.any() else 999)
    d = np.array(d); return float((np.abs(d) <= 1).mean()), float((d == 0).mean()), float((d == 999).mean()), float((d < -1).mean()), float(((d > 1) & (d < 999)).mean())
rows = []
for key in E[0]['G']:
    best = max(grid, key=lambda c: score(key, c)[0]); s = score(key, best)
    rows.append({'나누는 기준': key[0], '창(초)': int(key[1]), '조건': key[2], '최적 임계': round(float(best), 4), '±1행': s[0], '정확': s[1], '미발화': s[2], '2행 이상 빠름': s[3], '2행 이상 늦음': s[4]})
d = pd.DataFrame(rows).sort_values('±1행', ascending=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 80)
print(d.head(14).round(2).to_string(index=False))
top = d.iloc[0]; key = (top['나누는 기준'], float(top['창(초)']), top['조건'])
print('\n구성별 ±1행 일치율: 가이드 그대로(전체 범위, 20초, 0.02) vs 최고 후보', key, top['최적 임계'])
for c in sorted({x['cfg'] for x in E}):
    sub = [x for x in E if x['cfg'] == c]
    print(f"  {c:9s} n={len(sub):2d} | 가이드 그대로 {score(('전체 범위', 20.0, '차이'), 0.02, sub)[0]:.0%} | 최고 후보 {score(key, top['최적 임계'], sub)[0]:.0%} | 구성별 최적 임계로는 {max(score(key, cc, sub)[0] for cc in grid):.0%}")
