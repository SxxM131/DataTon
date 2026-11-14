"""[라운드 2] 시작 행(evalp 문맥의 단계 4/5/6 첫 행)에서 CNN 단독 정답률, 테이블이 틀린 시작 행에서의 CNN 정답률."""
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
pt = T.pred_r0.to_numpy().astype(int); y = evalp.label.to_numpy()
ON = L.onset_rows(evalp)
for tag in sys.argv[1:]:
    with np.load(L.ICACHE / f'cnn_{tag}.npz') as z:
        Pc = np.zeros((len(evalp), 6))
        for k in range(5):
            m = fold_of[pos] == k
            Pc[m] = z[f'out{k}_eval'][pos[m]]
    pc = Pc.argmax(1) + 1
    s = f'{tag:14s}'
    for st, r in ON.items():
        miss = r[pt[r] != y[r]]
        s += (f' | 단계{st} 시작 행 n={len(r)}: tab 정답 {(pt[r] == y[r]).sum()}, CNN 정답 {(pc[r] == y[r]).sum()}, tab 오답 {len(miss)}행 중 CNN 정답 {(pc[miss] == y[miss]).sum()}'
              f' (그 행의 평균 P_cnn(정답) {Pc[miss, st - 1].mean():.2f}), tab 정답 행 중 CNN 오답 {((pt[r] == y[r]) & (pc[r] != y[r])).sum()}')
    print(s)
