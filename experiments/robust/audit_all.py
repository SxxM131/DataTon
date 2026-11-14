"""전환 5개(1→2 … 5→6) 전수 점검: (1) 모델의 전환 시점 오차 분포, (2) 가이드 규칙에서 나오는 필요·충분 조건의 위반 여부와 정답.

기준 예측: 현재 최종 구성의 테이블 부분 (기본 모델 + 2·3 재판정 + 5단계 근거 확인), 공통 검증 3회 OOF.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
R = HERE.parent / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
y = evalp.label.to_numpy(int)
P = [M.gate_stage5(M.redecide_23(A[f'pred_r{r}'].to_numpy(int), B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy()), evalp) for r in range(3)]
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
order = evalp.assign(_i=np.arange(len(evalp))).sort_values(['experiment_id', 'relative_time_s'])
groups = [g._i.to_numpy() for _, g in order.groupby('experiment_id', sort=False)]

print('== (1) 전환 시점 오차: (모델이 처음 k+1 이상으로 예측한 행) − (실제 전환 행), 실험×3회 ==')
rows = []
for k in range(1, 6):
    d = []
    for ix in groups:
        if not (y[ix] > k).any() or not (y[ix] == k).any():
            continue
        i_true = int(np.argmax(y[ix] > k))
        for p in P:
            pp = p[ix]
            d.append(int(np.argmax(pp > k)) - i_true if (pp > k).any() else 999)
    d = np.array(d)
    rows.append({'전환': f'{k}→{k+1}', '실험×회': len(d), '정확': f'{(d == 0).mean():.0%}', '±1행': f'{(np.abs(d) <= 1).mean():.0%}', '2행 이상 빠름': f'{(d <= -2).mean():.0%}',
                 '2행 이상 늦음': f'{((d >= 2) & (d < 999)).mean():.0%}', '10행 이상 빠름': f'{(d <= -10).mean():.0%}', '10행 이상 늦음': f'{((d >= 10) & (d < 999)).mean():.0%}', '끝까지 못 넘김': f'{(d == 999).mean():.0%}'})
print(pd.DataFrame(rows).to_string(index=False))

# ---- 과거만으로 계산하는 근거 신호 ----
Rf = M.rule_features(evalp)
t1cm = np.zeros(len(evalp)); S = (evalp.meas_t1 + evalp.meas_t2).to_numpy(float); Scm = np.zeros(len(evalp)); Sprev = np.full(len(evalp), np.nan)
for ix in groups:
    t1cm[ix] = np.maximum.accumulate(evalp.meas_t1.to_numpy(float)[ix]); Scm[ix] = np.maximum.accumulate(S[ix]); Sprev[ix[1:]] = S[ix[:-1]]
vdrop = Rf.r_since_vdrop.to_numpy() >= 0
newmax = S >= Scm - 1e-9
rise = S - Sprev

print('\n== (2) 라벨 자체가 조건을 지키는가 (Test식 행 31,784개) ==')
chk = {'1단계인데 t1 누적최대 ≥ 45': (y == 1) & (t1cm >= 45), '2단계 이상인데 t1 누적최대 < 45': (y >= 2) & (t1cm < 45),
       '3단계 이상인데 t1 누적최대 < 50': (y >= 3) & (t1cm < 50), '2단계인데 t1 누적최대 > 90': (y == 2) & (t1cm > 90), '2단계인데 t1 누적최대 > 100': (y == 2) & (t1cm > 100),
       '3단계 이하인데 전압 급락 이벤트 있음': (y <= 3) & vdrop, '4단계 이상인데 전압 급락 이벤트 없음': (y >= 4) & ~vdrop,
       '6단계인데 지금이 새 최고점이고 직전보다 20℃ 넘게 오름': (y == 6) & newmax & (rise > 20), '5단계 이하인데 최고점보다 50℃ 넘게 내려옴': (y <= 5) & (y >= 4) & (S < Scm - 50)}
for k, m in chk.items():
    print(f'  {k:40s} {int(m.sum()):6d}행, 실험 {len(set(evalp.experiment_id[m]))}개')

print('\n== (3) 모델 예측이 조건을 어긴 행 (3회 평균)과 그 행의 정답 ==')
def rep(name, mask_fn, fix_fn):
    n = np.mean([mask_fn(p).sum() for p in P]); ok_now = np.mean([(mask_fn(p) & (p == y)).sum() for p in P]); ok_fix = np.mean([(mask_fn(p) & (fix_fn(p) == y)).sum() for p in P])
    g = pd.DataFrame({f'pred_r{r}': np.where(mask_fn(P[r]), fix_fn(P[r]), P[r]) for r in range(3)}, index=evalp.index)
    b = pd.DataFrame({f'pred_r{r}': P[r] for r in range(3)}, index=evalp.index)
    c = cv.compare(evalp, g, b, n_boot=2000) if n > 0 else {'diff': 0.0, 'p_a_better': float('nan')}
    print(f'  {name:46s} 위반 {n:6.1f}행 | 지금 맞음 {ok_now:5.1f} / 고치면 맞음 {ok_fix:5.1f} | 고쳤을 때 CV {c["diff"]:+.4f} (p {c["p_a_better"]:.2f})')
    return g
rep('1 예측인데 t1 누적최대 ≥ 45 → 2', lambda p: (p == 1) & (t1cm >= 45), lambda p: np.full(len(p), 2))
rep('2 이상 예측인데 t1 누적최대 < 45 → 1', lambda p: (p >= 2) & (t1cm < 45), lambda p: np.full(len(p), 1))
rep('3 이상 예측인데 t1 누적최대 < 50 → 2', lambda p: (p >= 3) & (t1cm < 50), lambda p: np.full(len(p), 2))
rep('2 예측인데 t1 누적최대 > 90 → 3', lambda p: (p == 2) & (t1cm > 90), lambda p: np.full(len(p), 3))
rep('2 예측인데 t1 누적최대 > 100 → 3', lambda p: (p == 2) & (t1cm > 100), lambda p: np.full(len(p), 3))
rep('3 이하 예측인데 전압 급락 이벤트 있음 → 4', lambda p: (p <= 3) & vdrop, lambda p: np.full(len(p), 4))
rep('4 이상 예측인데 전압 급락 이벤트 없음 → 3', lambda p: (p >= 4) & ~vdrop, lambda p: np.full(len(p), 3))
for d in (5, 10, 20, 40):
    rep(f'6 예측인데 새 최고점이고 직전보다 {d}℃ 넘게 오름 → 5', lambda p, d=d: (p == 6) & newmax & (rise > d), lambda p: np.full(len(p), 5))
for d in (20, 50, 100):
    rep(f'4·5 예측인데 최고점보다 {d}℃ 넘게 내려옴 → 6', lambda p, d=d: ((p == 5) | (p == 4)) & (S < Scm - d) & vdrop, lambda p: np.full(len(p), 6))
rep('단계가 뒤로 감(직전 행보다 낮음) → 직전 값 유지', lambda p: np.concatenate([[False], False * p[1:]]) if False else _back(p), lambda p: _hold(p)) if False else None
