"""3단계 규칙의 '정규화' 기준 식별: 어떤 기준으로 나눠야 '20초 뒤 상승 ≥ 임계'의 첫 지점이 라벨과 맞는가. (학습 라벨 분석, 미래 값 사용 = 분석 전용)

후보 기준 N (실험별 스칼라): 압력 범위(max−min), 최댓값, 최댓값−초기값, 초기값, 초기값 대비 변화율의 범위 등.
규칙: 온도 50~90℃에서 p(t+w) − p(t) ≥ c·N 인 첫 t. 각 후보에 대해 c를 훑어 라벨 시작 행과 ±1행 이내로 맞는 실험 비율을 본다.
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
    g = g.sort_values('relative_time_s'); t = g.relative_time_s.to_numpy(float); p = g.meas_p.to_numpy(float); t1 = g.meas_t1.to_numpy(float); y = g.label.to_numpy(int)
    if not (y == 3).any():
        continue
    bc, cc, tm, soc = int(g.bat_class.iloc[0]), float(g.bat_cc.iloc[0]), int(g.tr_method.iloc[0]), int(g.soc_status.iloc[0])
    special = (bc == 3 and cc == 280 and tm == 1) or (bc == 2 and cc == 73 and tm == 1 and soc == 1)
    i3 = int(np.argmax(y == 3)); p0 = p[0]; ap0 = max(abs(p0), 1e-3)
    x = (p - p0) / ap0                                   # 초기값 대비 변화율
    N = {'범위(max−min)': p.max() - p.min(), '최댓값': p.max(), '최댓값−초기값': max(p.max() - p0, 1e-6), '초기값': ap0, '|p−p0| 최대': np.abs(p - p0).max(),
         '상수 1': 1.0, '표준편차': p.std(), '3단계 이전 범위': max(p[:i3 + 1].max() - p[:i3 + 1].min(), 1e-6), '끝값−초기값': max(abs(p[-1] - p0), 1e-6),
         '변화율 범위×1': x.max() - x.min()}
    F = {}
    for w in (10.0, 15.0, 20.0, 25.0, 30.0):
        j = np.searchsorted(t, t + w, side='left'); ok = j < len(t); f = np.full(len(t), np.nan); f[ok] = p[j[ok]] - p[ok]; F[w] = f
    E.append({'cfg': f'{bc}/{cc:g}/{tm}', 'special': special, 'i3': i3, 'gate': (t1 >= 50) & (t1 <= 90), 'N': N, 'F': F, 'x': x, 'p0': ap0, 'n': len(t)})
print('실험', len(E), '개 (특수 구성', sum(e['special'] for e in E), '개 제외하고 평가)')
use = [e for e in E if not e['special']]


def score(key, c, w=20.0, relx=False):
    d = []
    for e in use:
        f = e['F'][w] / e['p0'] if relx else e['F'][w]
        m = e['gate'] & (np.nan_to_num(f, nan=-np.inf) >= c * e['N'][key])
        d.append(int(np.argmax(m)) - e['i3'] if m.any() else 999)
    d = np.array(d)
    return (np.abs(d) <= 0).mean(), (np.abs(d) <= 1).mean(), (np.abs(d) <= 2).mean(), (d == 999).mean()


rows = []
for key in use[0]['N']:
    grid = np.exp(np.linspace(np.log(1e-4), np.log(5.0), 140))
    for w in (15.0, 20.0, 25.0, 30.0):
        best = max(grid, key=lambda c: score(key, c, w)[1])
        s = score(key, best, w)
        rows.append({'기준': key, '창(초)': w, '최적 임계': round(float(best), 4), '정확': f'{s[0]:.0%}', '±1행': f'{s[1]:.0%}', '±2행': f'{s[2]:.0%}', '미발화': f'{s[3]:.0%}', '_': s[1]})
d = pd.DataFrame(rows).sort_values('_', ascending=False).drop(columns='_')
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 60)
print(d.head(16).to_string(index=False))
print('\n가이드 그대로 (범위, 20초, 0.02):', [f'{v:.0%}' for v in score('범위(max−min)', 0.02)])
# 구성별: 가이드 그대로 vs 가장 좋은 후보
best = d.iloc[0]
print('\n구성별 ±1행 일치율:')
for c in sorted({e['cfg'] for e in use}):
    sub = [e for e in use if e['cfg'] == c]
    def sc(key, cc_, w):
        dd = []
        for e in sub:
            m = e['gate'] & (np.nan_to_num(e['F'][w], nan=-np.inf) >= cc_ * e['N'][key]); dd.append(int(np.argmax(m)) - e['i3'] if m.any() else 999)
        return float((np.abs(np.array(dd)) <= 1).mean())
    print(f"  {c:10s} n={len(sub):2d} | 가이드 그대로 {sc('범위(max−min)', 0.02, 20.0):.0%} | 최적 후보({best['기준']}, {best['창(초)']:.0f}초, {best['최적 임계']}) {sc(best['기준'], best['최적 임계'], best['창(초)']):.0%}")
