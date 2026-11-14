"""[라운드 2] 단계 4/5/6 '첫 예측 행'이 실제 시작 행보다 몇 행(evalp 문맥) 빠른가/늦은가: 테이블 vs CNN vs 후기 융합."""
import sys
import numpy as np
import pandas as pd
import imglib as L
from imglib import cv

pool, evalp = L.load_all()
pos = pool.index.get_indexer(evalp.index)
fold_of = np.full(len(pool), -1)
for f in cv.make_folds(pool, n_repeats=1):
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
T = pd.read_pickle(L.HERE.parent / 'results/oof/tab_all_nocum_r3.pkl').reindex(evalp.index)
Pt = T[[f'p{k}_r0' for k in range(1, 7)]].to_numpy(float)
y = evalp.label.to_numpy(); t = evalp.relative_time_s.to_numpy()
groups = [idx[np.argsort(t[idx], kind='mergesort')] for idx in evalp.groupby('experiment_id', sort=False).indices.values()]


def cnn(tag):
    with np.load(L.ICACHE / f'cnn_{tag}.npz') as z:
        Pc = np.zeros((len(evalp), 6))
        for k in range(5):
            m = fold_of[pos] == k
            Pc[m] = z[f'out{k}_eval'][pos[m]]
    return Pc


def offsets(pred, s):
    off = []
    for idx in groups:
        if not (y[idx] == s).any():
            continue
        o = np.argmax(y[idx] == s); hit = pred[idx] >= s
        off.append((np.argmax(hit) - o) if hit.any() else 99)
    off = np.array(off)
    return f'정확 {(off == 0).sum()}, 1행 빠름 {(off == -1).sum()}, 2행+ 빠름 {(off <= -2).sum()}, 1행 늦음 {(off == 1).sum()}, 2행+ 늦음 {(off >= 2).sum()}'


models = {'tab_all_nocum_r3': Pt.argmax(1) + 1}
for tag in sys.argv[1:]:
    Pc = cnn(tag)
    models[f'CNN {tag}'] = Pc.argmax(1) + 1
    models[f'융합 w=0.2 {tag}'] = (np.log(Pt + 1e-9) * .8 + np.log(Pc + 1e-6) * .2).argmax(1) + 1
for s in (4, 5, 6):
    print(f'--- 단계 ≥{s} 첫 예측 행 vs 실제 단계 {s} 시작 행')
    for k, p in models.items():
        print(f'  {k:28s} {offsets(p, s)}')
