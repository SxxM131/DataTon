"""3단계 가설 검증: '표면 온도 50℃ 이상(게이트) 그리고 X가 임계값을 넘는 첫 행'.

세션 D의 스캔은 3단계 이전 모든 행에서 X가 임계 미만이어야 했다. 여기서는 게이트(t1>=50)를 통과한 행만 본다.
각 후보 X에 대해: (게이트 통과 후 3단계 시작 전 행들의 최댓값) < θ <= (3단계 시작 행의 값) 을 만족하는 실험 수가 최대가 되는 θ를 찾는다.
tol=1 은 라벨이 센서보다 1행 앞서는 경우(시작 다음 행에서 넘음)도 인정.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent.parent / 'common')]
import cv  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light').sort_values(['experiment_id', 'relative_time_s']).reset_index(drop=True)
g = pool.groupby('experiment_id', sort=False)
RAW = ['meas_v', 'meas_t1', 'meas_t2', 'meas_p', 'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o',
       'meas_no', 'meas_no2', 'meas_so2', 'atmp_t', 'thermal_t', 'indu_t']
C = {}
for c in RAW:
    x = pool[c]
    C[c] = x
    first = g[c].transform('first')
    C[f'{c}-first'] = x - first
    C[f'{c}/first'] = x / first.replace(0, np.nan)
    C[f'{c}.cummax'] = g[c].cummax()
    for k in (1, 2, 3, 6, 12, 24):
        C[f'{c}.d{k*5}'] = x - g[c].shift(k)
    for k in (3, 6, 12):
        C[f'{c}.mean{k*5}'] = g[c].transform(lambda s, k=k: s.rolling(k, min_periods=1).mean())
        C[f'{c}.max{k*5}'] = g[c].transform(lambda s, k=k: s.rolling(k, min_periods=1).max())
t1cm = g['meas_t1'].cummax()
C['t1-t2'] = pool.meas_t1 - pool.meas_t2
C['t1+t2'] = pool.meas_t1 + pool.meas_t2
C['t2-atmp'] = pool.meas_t2 - pool.atmp_t
C['t1-atmp'] = pool.meas_t1 - pool.atmp_t
C['thermal-atmp'] = pool.thermal_t - pool.atmp_t
C['indu-t1'] = pool.indu_t - pool.meas_t1
C['time'] = pool.relative_time_s
# t1이 45 / 50에 도달한 시점의 값 대비
for thr in (45, 50):
    hit = (t1cm >= thr)
    for c in ['meas_p', 'meas_t2', 'meas_v', 'thermal_t', 'indu_t', 'relative_time_s']:
        ref = pool[c].where(hit).groupby(pool.experiment_id, sort=False).transform('first')
        C[f'{c}-at{thr}'] = pool[c] - ref
gas = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
C['gas_sum'] = pool[gas].sum(axis=1)
C['gas_max'] = pool[gas].max(axis=1)
X = pd.DataFrame(C)
gate = (t1cm >= 50).to_numpy()
lab = pool.label.to_numpy()
eid = pool.experiment_id.to_numpy()
exps = pool.experiment_id.unique()
onset = {e: np.where((eid == e) & (lab >= 3))[0][0] for e in exps}
pre = {e: np.where((eid == e) & gate & (lab < 3))[0] for e in exps}
n_gate_first = sum(len(pre[e]) == 0 for e in exps)
print(f'실험 {len(exps)}개. 게이트 통과 첫 행 = 3단계 시작인 실험 {n_gate_first}개 (이들은 어떤 X든 θ<=X(onset)이면 일치)')


def best(col, sign, tol):
    v = X[col].to_numpy(float) * sign
    lo, hi = [], []
    for e in exps:
        i = onset[e]
        p = pre[e]
        m = np.nanmax(v[p]) if len(p) and not np.all(np.isnan(v[p])) else -np.inf
        cand = v[i:i + tol + 1]
        cand = cand[(eid[i:i + tol + 1] == e)]
        h = np.nanmax(cand) if len(cand) and not np.all(np.isnan(cand)) else np.nan
        lo.append(m); hi.append(h)
    lo, hi = np.array(lo), np.array(hi)
    ok = ~np.isnan(hi)
    ths = np.unique(np.concatenate([hi[ok], lo[ok & np.isfinite(lo)] + 1e-9]))
    if len(ths) > 600:
        ths = np.quantile(ths, np.linspace(0, 1, 600))
    cnt = [(int(((lo < th) & (th <= hi) & ok).sum()), th) for th in ths]
    return max(cnt) if cnt else (0, np.nan)


rows = []
for col in X.columns:
    for sign in (1, -1):
        for tol in (0, 1):
            c, th = best(col, sign, tol)
            rows.append((col, '>=' if sign > 0 else '<=', tol, c, th * sign))
r = pd.DataFrame(rows, columns=['X', '방향', '허용행', '일치 실험 수', 'θ']).sort_values('일치 실험 수', ascending=False)
pd.set_option('display.width', 200)
print(r[r['허용행'] == 0].head(15).to_string(index=False))
print()
print(r[r['허용행'] == 1].head(15).to_string(index=False))
