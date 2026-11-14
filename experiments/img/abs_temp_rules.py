"""절대 온도로 변환한 열화상 신호를 '규칙의 근거'로 쓸 수 있는가.

(A) 3단계 시작을 알리는가: 신호가 임계에 처음 도달한 시점으로 3단계 시작을 맞히면 얼마나 어긋나는가 (임계 = 같은 구성의 다른 실험들에서 본 시작 시점 값의 중앙값).
    센서(t1, t2, thermal_t)와 절대 온도 신호를 같은 방식으로 비교. 원본 5초 행, 오차는 초.
(B) 6단계 판정의 근거가 되는가: v9 최종 예측 위에 (B1) "6으로 판정했지만 이미지 신호가 아직 최고치를 갱신 중이면 5", (B2) "5로 판정했지만 이미지 신호가 최고치에서 일정 비율 내려왔으면 6"을 얹어 본다.
사용: python experiments/img/abs_temp_rules.py   (abs_temp2.py가 만든 신호 캐시 사용)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
F = pd.read_pickle(C / 'img' / 'abs_temp2_signals.pkl')
CLS = {1: '원통형', 2: '파우치', 3: '각형'}; TM = {1: '가열', 2: '과충전'}
name_of = lambda fr: (fr.bat_class.map(CLS) + ' ' + fr.bat_cc.map(lambda v: f'{v:g}Ah') + ' ' + fr.tr_method.map(TM))  # noqa: E731
TESTC = ['각형 280Ah 과충전', '각형 28Ah 가열', '원통형 4.8Ah 가열', '원통형 6Ah 가열', '파우치 39Ah 가열', '파우치 73Ah 과충전']

# ---------------- (A) 3단계 시작 ----------------
P = pool.assign(cfg=name_of(pool)).join(F); P['t1'] = P.meas_t1; P['t2'] = P.meas_t2
SIGS = ['t1', 't2', 'thermal_t', 'T_mean', 'T_center', 'T_p90', 'A50', 'A80']
onset, series = {}, {}
for ex, g in P.groupby('experiment_id', sort=False):
    g = g.sort_values('relative_time_s'); lab = g.label.to_numpy()
    if not (lab >= 3).any() or not (lab == 2).any():
        continue
    i3, i2 = int(np.argmax(lab >= 3)), int(np.argmax(lab >= 2))
    series[ex] = (g.cfg.iloc[0], g.relative_time_s.to_numpy(float), {s: g[s].to_numpy(float) for s in SIGS}, i2, i3)
rows = []
for cname in sorted({v[0] for v in series.values()}):
    exs = [k for k, v in series.items() if v[0] == cname]; r = {'구성': cname, 'Test': cname in TESTC, '실험': len(exs)}
    for s in SIGS:
        err = []
        for ex in exs:
            th = np.median([series[o][2][s][series[o][4]] for o in exs if o != ex]) if len(exs) > 1 else np.nan
            _, t, sg, i2, i3 = series[ex]; x = sg[s]; m = (np.arange(len(x)) >= i2) & (x >= th)       # 2단계에 들어선 뒤 임계에 처음 도달
            ip = int(np.argmax(m)) if m.any() else len(x) - 1; err.append(abs(t[ip] - t[i3]))
        r[s] = round(float(np.median(err)), 0)
    rows.append(r)
A = pd.DataFrame(rows); print('(A) 3단계 시작 시점 오차의 중앙값 (초). 작을수록 그 신호가 3단계 시작을 잘 알린다')
print(A.to_string(index=False))
m = A.Test; print('Test 구성 평균:', A[m][SIGS].mean().round(0).to_dict()); print('전체 구성 평균:', A[SIGS].mean().round(0).to_dict())

# ---------------- (B) 6단계 판정 근거 ----------------
y = evalp.label.to_numpy(int); e = evalp.experiment_id.to_numpy(); cfg = name_of(evalp).to_numpy(); TEST = np.isin(cfg, TESTC)
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
Ar = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); Br = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
base = [M.apply_rules(Ar[f'pred_r{r}'].to_numpy(int), Br[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(), evalp) for r in range(3)]
Fe = F.loc[evalp.index].assign(thermal_t=evalp.thermal_t.to_numpy(float)); order = np.lexsort((evalp.relative_time_s.to_numpy(float), e)); eo = e[order]
def cummax(x):
    out = np.empty(len(x)); out[order] = pd.Series(x[order]).groupby(eo).cummax().to_numpy(); return out
O = lambda P_: pd.DataFrame({f'pred_r{r}': P_[r] for r in range(3)}, index=evalp.index)  # noqa: E731
mf = lambda P_, m=None: float(np.mean([f1_score(y if m is None else y[m], p if m is None else p[m], labels=range(1, 7), average='macro') for p in P_]))  # noqa: E731
b0, b0t = mf(base), mf(base, TEST); print(f'\n(B) v9 최종: {b0:.4f} | Test 구성 {b0t:.4f}')
res = []
for s in ['thermal_t', 'T_mean', 'T_center', 'T_p90', 'A100', 'A150', 'A200']:
    x = Fe[s].to_numpy(float); cm = cummax(x); rising = x >= cm - 1e-9
    v = [np.where((p == 6) & rising & (cm > 0), 5, p) for p in base]; c = cv.compare(evalp, O(v), O(base), n_boot=1000)
    fx = np.mean([((p != q) & (q == y)).sum() for p, q in zip(base, v)]); br = np.mean([((p != q) & (p == y)).sum() for p, q in zip(base, v)])
    res.append({'규칙': f'B1 6→5: {s}가 아직 최고치', '점수': round(mf(v), 4), '차이': round(c['diff'], 4), '개선 확률': round(c['p_a_better'], 2), 'Test 구성 차이': round(mf(v, TEST) - b0t, 4), '고친 행': round(fx, 1), '망친 행': round(br, 1)})
    for frac in (0.1, 0.2, 0.3):
        dropped = (cm > 0) & (x <= (1 - frac) * cm)
        v = [np.where((p == 5) & dropped, 6, p) for p in base]; c = cv.compare(evalp, O(v), O(base), n_boot=1000)
        fx = np.mean([((p != q) & (q == y)).sum() for p, q in zip(base, v)]); br = np.mean([((p != q) & (p == y)).sum() for p, q in zip(base, v)])
        res.append({'규칙': f'B2 5→6: {s}가 최고치에서 {int(frac * 100)}% 하강', '점수': round(mf(v), 4), '차이': round(c['diff'], 4), '개선 확률': round(c['p_a_better'], 2), 'Test 구성 차이': round(mf(v, TEST) - b0t, 4), '고친 행': round(fx, 1), '망친 행': round(br, 1)})
T = pd.DataFrame(res).sort_values('차이', ascending=False)
print('B1 (6 → 5로 되돌리기)'); print(T[T['규칙'].str.startswith('B1')].to_string(index=False))
print('B2 (5 → 6으로 올리기) 가장 나은 5개'); print(T[T['규칙'].str.startswith('B2')].head(5).to_string(index=False))
