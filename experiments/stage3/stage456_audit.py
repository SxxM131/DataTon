"""4~6단계 경계: 구성별 오류 행과, 가이드의 5단계 규칙(구성별 종류)을 과거만으로 근사했을 때 라벨과 맞는 정도. 분석용."""
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
y = evalp.label.to_numpy(int)
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
rows = {}
for lo, hi in ((3, 4), (4, 5), (5, 6)):
    m = np.mean([(np.isin(y, (lo, hi)) & np.isin(A[f'pred_r{r}'].to_numpy(int), (lo, hi)) & (A[f'pred_r{r}'].to_numpy(int) != y)) for r in range(3)], axis=0)
    rows[f'{lo}↔{hi}'] = pd.Series(m).groupby(cfg).sum().round(0).astype(int)
d = pd.DataFrame(rows)
d['행(4)'] = pd.Series(y == 4).groupby(cfg).sum(); d['행(5)'] = pd.Series(y == 5).groupby(cfg).sum(); d['행(6)'] = pd.Series(y == 6).groupby(cfg).sum()
d['실험'] = evalp.groupby(cfg).experiment_id.nunique()
print(d.to_string())

# --- 5단계 시작: 가이드 규칙의 인과 근사 vs 라벨 (원본 5초 행) ---
DOC5 = {'3/280.0/1': 'v0', '2/39.0/1': 'v0', '2/73.0/1': 'v0', '2/73.0/2': 'v0', '1/4.8/1': 'vdd', '1/6.0/1': 'vdd', '3/28.0/1': 'tr2', '3/280.0/2': 'tr2', '2/39.0/2': 'tr2'}
out = []
pcfg = (pool.bat_class.astype(int).astype(str) + '/' + pool.bat_cc.astype(str) + '/' + pool.tr_method.astype(int).astype(str))
for e, g in pool.groupby('experiment_id', sort=True):
    g = g.sort_values('relative_time_s'); c = pcfg[g.index[0]]
    t = g.relative_time_s.to_numpy(float); yy = g.label.to_numpy(int); v = g.meas_v.to_numpy(float); t1 = g.meas_t1.to_numpy(float)
    if not (yy == 5).any():
        continue
    i5 = int(np.argmax(yy == 5)); n = len(t)
    def lagv(x, w):
        j = np.searchsorted(t, t - w, side='right') - 1; ok = j >= 0
        o = np.full(n, np.nan); o[ok] = x[j[ok]]; return o
    ev = {}
    m = v <= 0.005; ev['v0'] = int(np.argmax(m)) if m.any() else -1
    vmax = np.maximum.accumulate(v)
    drop15 = np.nanmax(np.stack([lagv(v, w) for w in (5.0, 10.0, 15.0)]), 0) - v           # 지난 15초 안의 최고 전압 대비 하강량
    big = np.nan_to_num(drop15, nan=0.0) >= 0.25 * np.maximum(vmax, 1e-6)
    # 하강 '사건'을 묶는다: 15초 넘게 떨어져 있으면 다른 사건
    idx = np.flatnonzero(big); starts = [idx[0]] + [b for a, b in zip(idx[:-1], idx[1:]) if t[b] - t[a] > 15] if len(idx) else []
    ev['vdd'] = int(starts[1]) if len(starts) > 1 else -1
    ev['vd1'] = int(starts[0]) if len(starts) > 0 else -1
    for thr in (2.0, 4.0, 8.0):                                                             # 10초 동안 온도 상승 (℃/10초)
        mm = np.nan_to_num(t1 - lagv(t1, 10.0), nan=0.0) >= thr * 10
        ev[f'tr{thr:g}'] = int(np.argmax(mm)) if mm.any() else -1
    r = {'cfg': c, 'doc': DOC5.get(c, '?'), 'i4': int(np.argmax(yy == 4)) if (yy == 4).any() else -1, 'i5': i5}
    r.update({k: (ix - i5 if ix >= 0 else np.nan) for k, ix in ev.items()})
    out.append(r)
o = pd.DataFrame(out)
pd.set_option('display.width', 250)
def summ(x):
    x = x.dropna(); return f"{int((x == 0).sum())}/{int((x == 1).sum())}/{int((x == 2).sum())} 조기 {int((x < 0).sum())} 늦음3+ {int((x > 2).sum())}"
print('\n5단계 시작 행 대비 규칙 발화 위치 (정확 / +1행 / +2행, 조기, 3행 이상 늦음; 미발화는 제외):')
print(o.groupby(['cfg', 'doc']).agg(n=('i5', 'size'), **{k: (k, summ) for k in ('v0', 'vdd', 'vd1', 'tr2', 'tr4', 'tr8')}).to_string())
