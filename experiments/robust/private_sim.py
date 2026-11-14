"""Private 하방 위험 모의: Test와 같은 구성으로 실험 11개를 뽑고 그중 7~8개를 Private으로 삼아 채점 (교차검증 OOF, 3회 반복을 섞어 사용).

후보별 평균뿐 아니라 하위 5%·10%(운이 나쁠 때)와, 'Public이 높게 나온 경우'의 Private 분포를 본다.
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
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool); y = evalp.label.to_numpy(int)
R = HERE.parent / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
z = np.load(C / 'robust_img10_oof.npz', allow_pickle=True)
ca = [A[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(float) for r in range(3)]; cb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(float) for r in range(3)]
na = [z['PA'][r].astype(float) for r in range(3)]; nb = [z['PB'][r].astype(float) for r in range(3)]
v7 = lambda a, pb: M.hold_stage(M.floor_stage4(M.gate_stage5(M.gate_stage6(M.redecide_23(a, pb), evalp), evalp), evalp), evalp)  # noqa: E731
CAND = {'규칙 없음 (기본 모델)': [M.to_label(ca[r]) for r in range(3)],
        'v2 (2·3 재판정만)': [M.redecide_23(M.to_label(ca[r]), cb[r]) for r in range(3)],
        'v7': [v7(M.to_label(ca[r]), cb[r]) for r in range(3)],
        'v8 (현재)': [M.apply_rules(M.to_label(ca[r]), cb[r], evalp) for r in range(3)],
        '팀원안(img10) + v8 규칙': [M.apply_rules(M.to_label(na[r]), nb[r], evalp) for r in range(3)],
        '두 모델 평균 + v8 규칙': [M.apply_rules(M.to_label((ca[r] + na[r]) / 2), (cb[r] + nb[r]) / 2, evalp) for r in range(3)]}
key = list(zip(evalp.bat_class.astype(int), evalp.bat_cc.astype(float), evalp.tr_method.astype(int))); e = evalp.experiment_id.to_numpy()
exps = pd.DataFrame({'e': e, 'k': key}).drop_duplicates('e'); by = {k: g.e.to_numpy() for k, g in exps.groupby('k')}
rows = {x: np.flatnonzero(e == x) for x in exps.e}
TEST = {(1, 4.8, 1): 4, (1, 6.0, 1): 1, (2, 39.0, 1): 1, (2, 73.0, 2): 1, (3, 28.0, 1): 2, (3, 280.0, 2): 2}
rng = np.random.default_rng(0); N = 4000
draws = []
for _ in range(N):
    t = np.concatenate([rng.choice(by[k], n, replace=False) for k, n in TEST.items()]); npub = rng.choice([3, 4]); pub = rng.choice(t, npub, replace=False); pri = np.setdiff1d(t, pub)
    draws.append((int(rng.integers(3)), np.concatenate([rows[x] for x in pub]), np.concatenate([rows[x] for x in pri])))
f1 = lambda idx, p: f1_score(y[idx], p[idx], labels=range(1, 7), average='macro', zero_division=0)  # noqa: E731
res = {}
for name, P in CAND.items():
    res[name] = np.array([(f1(ip, P[r]), f1(iq, P[r])) for r, ip, iq in draws])
print(f"{'후보':24s} | Private 평균 | 표준편차 | 하위 5% | 하위 10% | 최저 | Public 평균")
for name, a in res.items():
    pr = a[:, 1]; print(f"{name:24s} | {pr.mean():.4f} | {pr.std():.4f} | {np.quantile(pr, .05):.4f} | {np.quantile(pr, .10):.4f} | {pr.min():.4f} | {a[:, 0].mean():.4f}")
ref = res['v8 (현재)']
print('\n같은 뽑기에서 v8 대비 Private 차이 (양수 = 그 후보가 높음)')
for name, a in res.items():
    if name != 'v8 (현재)':
        d = a[:, 1] - ref[:, 1]; print(f"  {name:24s} 평균 {d.mean():+.4f} | v8보다 높은 비율 {(d > 0).mean():.0%} | 0.01 넘게 낮은 비율 {(d < -0.01).mean():.1%} | 0.01 넘게 높은 비율 {(d > 0.01).mean():.1%}")
a = ref; hi = a[:, 0] >= 0.94
print(f"\nv8: Public ≥ 0.94 인 경우는 전체의 {hi.mean():.0%}. 그때 Private 평균 {a[hi, 1].mean():.4f}, 하위 5% {np.quantile(a[hi, 1], .05):.4f}, 하위 10% {np.quantile(a[hi, 1], .10):.4f}")
print(f"    그때 (Public − Private) 평균 {(a[hi, 0] - a[hi, 1]).mean():+.4f}, Private이 Public보다 0.03 넘게 낮을 확률 {((a[hi, 0] - a[hi, 1]) > 0.03).mean():.0%}, 0.05 넘게 낮을 확률 {((a[hi, 0] - a[hi, 1]) > 0.05).mean():.0%}")
print(f"v8 전체: Private < 0.90 일 확률 {(a[:, 1] < 0.90).mean():.0%}, < 0.88 일 확률 {(a[:, 1] < 0.88).mean():.0%}, ≥ 0.93 일 확률 {(a[:, 1] >= 0.93).mean():.0%}")
