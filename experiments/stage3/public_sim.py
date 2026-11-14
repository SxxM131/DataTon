"""Public(실험 3~4개)에서 우리 모델과 '문서 규칙 그대로(규정 위반)' 모델이 얼마나 벌어질 수 있는지 CV 예측으로 모의. 분석 전용.

모의: Test와 같은 구성 비율로 실험 11개를 뽑고 → 그중 3개 또는 4개를 Public, 나머지를 Private으로 나눠 같은 부분집합에서 세 모델을 채점.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
C = Path.home() / 'jaehyeok' / 'cache'; R = HERE.parent / 'results' / 'oof'
y = evalp.label.to_numpy(int)
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
a = A['pred_r0'].to_numpy(int); b = np.where(B['p3_r0'].to_numpy() > B['p2_r0'].to_numpy(), 3, 2)
v2 = np.where((a == 2) | (a == 3), b, a)
nc = pd.read_pickle(C / 'noncausal_doc_oof_r0.pkl').reindex(evalp.index)['pred_r0'].to_numpy(int)
d = pd.read_pickle(C / 'seq' / 'proba_lr_r0.pkl'); assert (d['index'] == evalp.index.to_numpy()).all()
lr = d['proba'].argmax(1) + 1
key = list(zip(evalp.bat_class.astype(int), evalp.bat_cc.astype(float), evalp.tr_method.astype(int)))
e = evalp.experiment_id.to_numpy()
exps = pd.DataFrame({'e': e, 'k': key}).drop_duplicates('e')
by = {k: g.e.to_numpy() for k, g in exps.groupby('k')}
rows = {x: np.flatnonzero(e == x) for x in exps.e}
TEST = {(1, 4.8, 1): 4, (1, 6.0, 1): 1, (2, 39.0, 1): 1, (2, 73.0, 2): 1, (3, 28.0, 1): 2, (3, 280.0, 2): 2}
f1 = lambda idx, p: f1_score(y[idx], p[idx], labels=range(1, 7), average='macro', zero_division=0)  # noqa: E731
rng = np.random.default_rng(0); out = []
for it in range(6000):
    t = np.concatenate([rng.choice(by[k], n, replace=False) for k, n in TEST.items()])
    pub = rng.choice(t, rng.choice([3, 4]), replace=False); pri = np.setdiff1d(t, pub)
    ip = np.concatenate([rows[x] for x in pub]); iq = np.concatenate([rows[x] for x in pri])
    out.append((f1(ip, v2), f1(ip, nc), f1(ip, lr), f1(iq, v2), f1(iq, nc), f1(iq, lr)))
o = pd.DataFrame(out, columns=['pub_v2', 'pub_nc', 'pub_lr', 'pri_v2', 'pri_nc', 'pri_lr'])
q = lambda s: f"{s.mean():.3f} (5~95%: {s.quantile(.05):.3f}~{s.quantile(.95):.3f})"  # noqa: E731
print('Public 모의 점수   우리 v2:', q(o.pub_v2), '| 규칙 그대로:', q(o.pub_nc), '| 베이스라인 LR:', q(o.pub_lr))
print('Private 모의 점수  우리 v2:', q(o.pri_v2), '| 규칙 그대로:', q(o.pri_nc), '| 베이스라인 LR:', q(o.pri_lr))
g = o.pub_nc - o.pub_v2
print(f'같은 Public에서 (규칙 그대로 − 우리): 평균 {g.mean():+.3f}, 5~95% {g.quantile(.05):+.3f}~{g.quantile(.95):+.3f}, 0.10 이상 벌어질 확률 {(g >= 0.10).mean():.1%}')
print(f'우리 Public ≤ 0.835일 확률 {(o.pub_v2 <= 0.835).mean():.1%} | 그때 LR ≥ 0.85일 확률 {((o.pub_v2 <= 0.835) & (o.pub_lr >= 0.85)).mean():.1%}')
m = (o.pub_v2 > 0.81) & (o.pub_v2 < 0.86)
print(f'우리 Public이 0.81~0.86인 경우 ({m.mean():.1%}): 같은 Public에서 규칙 그대로 {q(o.pub_nc[m])}, LR {q(o.pub_lr[m])}')
print(f'   그 경우 Private: 우리 v2 {q(o.pri_v2[m])} | 규칙 그대로 {q(o.pri_nc[m])} | LR {q(o.pri_lr[m])}')
m2 = m & (o.pub_lr >= 0.84)
print(f'우리 Public 0.81~0.86 이면서 LR Public ≥ 0.84인 경우 ({m2.mean():.1%}): Private 우리 v2 {q(o.pri_v2[m2])} | LR {q(o.pri_lr[m2])} | 우리가 LR보다 높을 확률 {(o.pri_v2[m2] > o.pri_lr[m2]).mean():.0%}')
