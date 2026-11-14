"""2·3단계 재판정을 이미지 융합과 어떻게 결합할지 비교 (rep 0, CNN OOF가 rep 0만 있음).

A = 기본 테이블 모델 OOF (rules_tab_nocum_r3), B = 압력 상승 피처를 더한 모델 OOF (st3_presrise_mask_r3), C = pairdiff CNN OOF.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

DATA = Path.home() / 'data' / 'battery_data_light'; ICACHE = Path.home() / 'jaehyeok' / 'cache' / 'img'
pool = cv.load_pool(DATA); evalp = cv.to_test_like(pool)
pos = pool.index.get_indexer(evalp.index)
fold_of = np.full(len(pool), -1)
for f in cv.make_folds(pool, n_repeats=1):
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
R = HERE.parent / 'results' / 'oof'
cols = [f'p{k}_r0' for k in range(1, 7)]
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index)[cols].to_numpy(float)
B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)[cols].to_numpy(float)
with np.load(ICACHE / 'cnn_pairdiff.npz') as z:
    C = np.zeros((len(evalp), 6))
    for k in range(5):
        m = fold_of[pos] == k
        C[m] = z[f'out{k}_eval'][pos[m]]


def fuse(pt, pc, w):
    zz = np.log(pt + 1e-9) * (1 - w) + np.log(pc + 1e-6) * w
    zz = zz - zz.max(1, keepdims=True)
    p = np.exp(zz)
    return p / p.sum(1, keepdims=True)


def redo(pred, Q):
    """pred가 2 또는 3인 행만 Q의 p2 vs p3로 다시 판정."""
    out = pred.copy()
    sel = (pred == 2) | (pred == 3)
    out[sel] = np.where(Q[sel, 2] > Q[sel, 1], 3, 2)
    return out


def O(pred):
    return pd.DataFrame({'pred_r0': pred.astype(int)}, index=evalp.index)


base = O(A.argmax(1) + 1)
V = {'A (테이블 기본)': A.argmax(1) + 1, 'A → 2·3 재판정(B)': redo(A.argmax(1) + 1, B), 'B 전체 교체': B.argmax(1) + 1}
for w in (0.1, 0.2):
    fa, fb = fuse(A, C, w), fuse(B, C, w)
    V[f'융합 w={w} (A+C)'] = fa.argmax(1) + 1
    V[f'융합 w={w} → 재판정(B 단독)'] = redo(fa.argmax(1) + 1, B)
    V[f'융합 w={w} → 재판정(B+C 융합)'] = redo(fa.argmax(1) + 1, fb)
    V[f'융합 w={w} (B+C) 전체 교체'] = fb.argmax(1) + 1
V['A → 재판정(B) 후 4~6만 융합 w=0.2'] = None
fa = fuse(A, C, 0.2).argmax(1) + 1
x = redo(A.argmax(1) + 1, B)
V['A → 재판정(B) 후 4~6만 융합 w=0.2'] = np.where((x >= 4) | (fa >= 4), fa, x)
v1 = O(V['융합 w=0.2 (A+C)'])
for name, pred in V.items():
    o = O(pred); r = cv.evaluate(evalp, o, n_boot=200); c = cv.compare(evalp, o, base, n_boot=3000); c2 = cv.compare(evalp, o, v1, n_boot=3000)
    print(f"{name:36s} CV {r['cv']:.4f} | long {r['long_exps']:.3f} short {r['short_exps']:.3f} | per-class {np.round(r['per_class'], 3).tolist()}"
          f" | vs A {c['diff']:+.4f} p {c['p_a_better']:.2f} | vs v1 {c2['diff']:+.4f} p {c2['p_a_better']:.2f}", flush=True)
