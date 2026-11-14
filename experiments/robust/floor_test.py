"""단계 최소 길이(하한) 제약의 공통 검증. 하한은 학습 fold 실험에서만 계산한다.

제약: 같은 실험에서 s단계에 들어선 뒤(최종 판정 기준) floor(구성, s)초가 지나기 전에는 s+1단계 이상으로 넘기지 않는다.
단계별·구성 묶음별로 따로 평가한다 (기준: 현재 최종 체인 apply_rules).
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
y = evalp.label.to_numpy(int); t = evalp.relative_time_s.to_numpy(float)
key = lambda f: (f.bat_class.astype(int).astype(str) + '/' + f.bat_cc.astype(str) + '/' + f.tr_method.astype(int).astype(str))  # noqa: E731
cfg = key(evalp).to_numpy(); e = evalp.experiment_id.to_numpy()
o = evalp.assign(_i=np.arange(len(evalp))).sort_values(['experiment_id', 'relative_time_s']); groups = {k: g._i.to_numpy() for k, g in o.groupby('experiment_id', sort=False)}
P = [M.apply_rules(A[f'pred_r{r}'].to_numpy(int), B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(), evalp) for r in range(3)]

# 실험별 단계 길이(초): 원본 5초 행의 라벨 기준 (다음 단계 시작 시각 − 이 단계 시작 시각)
dur = {}
for ex, g in pool.sort_values('relative_time_s').groupby('experiment_id'):
    tt = g.relative_time_s.to_numpy(float); lab = g.label.to_numpy(int); d = {}
    for s in range(2, 6):
        if (lab == s).any() and (lab > s).any():
            d[s] = tt[np.argmax(lab > s)] - tt[np.argmax(lab == s)]
    dur[ex] = d
pcfg = key(pool).groupby(pool.experiment_id).first()


def apply_floor(p, floors, stages, cfgs):
    q = p.copy()
    for ex, ix in groups.items():
        c = cfg[ix[0]]
        if c not in cfgs or c not in floors:
            continue
        x = q[ix]; tt = t[ix]
        for s in stages:
            fl = floors[c].get(s)
            if fl is None or not (x >= s).any():
                continue
            t_in = tt[np.argmax(x >= s)]
            x = np.where((x > s) & (tt - t_in < fl - 1e-9), s, x)
        q[ix] = x
    return q


folds = cv.make_folds(pool)
ALL = sorted(set(cfg)); CYL = ['1/4.8/1', '1/6.0/1']; TEST = ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2']
base = pd.DataFrame({f'pred_r{r}': P[r] for r in range(3)}, index=evalp.index)
for name, stages, cfgs, factor in [('원통형, 4단계 하한', (4,), CYL, 1.0), ('원통형, 4단계 하한 ×0.8', (4,), CYL, 0.8), ('전체 구성, 4단계 하한', (4,), ALL, 1.0), ('Test 구성, 4단계 하한', (4,), TEST, 1.0),
                                   ('전체 구성, 2단계 하한', (2,), ALL, 1.0), ('전체 구성, 3단계 하한', (3,), ALL, 1.0), ('전체 구성, 5단계 하한', (5,), ALL, 1.0), ('원통형, 5단계 하한', (5,), CYL, 1.0),
                                   ('전체 구성, 2·4·5단계 하한', (2, 4, 5), ALL, 1.0)]:
    Q = [p.copy() for p in P]
    for f in folds:
        tr_e = set(f['train_exps']); va = np.isin(e, f['valid_exps'])
        floors = {}
        for c in ALL:
            ds = [dur[x] for x in tr_e if pcfg[x] == c]
            floors[c] = {s: factor * min(d[s] for d in ds if s in d) for s in stages if sum(s in d for d in ds) >= 3}
        q = apply_floor(P[f['rep']], floors, stages, cfgs)
        Q[f['rep']][va] = q[va]
    g = pd.DataFrame({f'pred_r{r}': Q[r] for r in range(3)}, index=evalp.index)
    n = sum(int((Q[r] != P[r]).sum()) for r in range(3)); fx = sum(int(((Q[r] != P[r]) & (Q[r] == y)).sum()) for r in range(3)); br = sum(int(((Q[r] != P[r]) & (P[r] == y)).sum()) for r in range(3))
    c = cv.compare(evalp, g, base, n_boot=2000)
    print(f"{name:26s} 바뀐 행(3회 합) {n:4d} | 고쳐짐 {fx:4d} / 망가짐 {br:4d} | CV {c['diff']:+.4f} (p {c['p_a_better']:.2f})", flush=True)
print('\n구성별 4단계 길이(초, 학습 97개): ')
print(pd.DataFrame([{'cfg': pcfg[x], 'd4': d.get(4), 'd2': d.get(2), 'd5': d.get(5)} for x, d in dur.items()]).groupby('cfg').agg(n=('d4', 'count'), d4_min=('d4', 'min'), d4_5pct=('d4', lambda v: v.quantile(.05)), d4_med=('d4', 'median'), d2_min=('d2', 'min'), d5_min=('d5', 'min')).to_string())
