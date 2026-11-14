"""이미지 백본 비교: 같은 학습(현재 + 직전 프레임 + 차이 9ch)에서 백본만 바꾼 모델의 OOF를 최종 구성(융합 w=0.2 + 규칙 체인)에 넣어 비교.

사용: python experiments/img/eval_arch.py pairdiff vit_s128 vit_s224 ...   (~/jaehyeok/cache/img/cnn_{tag}.npz, 반복 0의 5-fold)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool); y = evalp.label.to_numpy(int)
pos = pool.index.get_indexer(evalp.index); fold_of = np.full(len(pool), -1)
for f in cv.make_folds(pool, n_repeats=1):
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
fo = fold_of[pos]
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'; cols = [f'p{k}_r0' for k in range(1, 7)]
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index)[cols].to_numpy(float)
B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)[cols].to_numpy(float)
O = lambda p: pd.DataFrame({'pred_r0': np.asarray(p).astype(int)}, index=evalp.index)  # noqa: E731
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])


def load(tag):
    with np.load(Path.home() / 'jaehyeok' / 'cache' / 'img' / f'cnn_{tag}.npz') as z:
        C = np.zeros((len(evalp), 6))
        for k in range(5):
            m = fo == k; C[m] = z[f'out{k}_eval'][pos[m]]
    return C


tab = M.apply_rules(M.to_label(A), B, evalp); ref = None
print(f'테이블 + 규칙 체인 (이미지 없음, 반복 0): {cv.macro_f1(y, tab):.4f} | Test 구성 {cv.macro_f1(y[TEST], tab[TEST]):.4f}')
for tag in sys.argv[1:]:
    C = load(tag); alone = M.to_label(C); s = f'{tag:10s} 이미지 단독 {cv.macro_f1(y, alone):.4f} {np.round(cv.evaluate(evalp, O(alone), n_boot=50)["per_class"], 2).tolist()}'
    for w in (0.2, 0.3):
        fin = M.apply_rules(M.to_label(M.fuse(A, C, w)), B, evalp); c = cv.compare(evalp, O(fin), O(tab), n_boot=2000)
        s += f' | 융합 w={w} + 규칙 {cv.macro_f1(y, fin):.4f} (이미지 없음 대비 {c["diff"]:+.4f}, p {c["p_a_better"]:.2f}; Test 구성 {cv.macro_f1(y[TEST], fin[TEST]):.4f}'
        if w == 0.2:
            if ref is None:
                ref = fin
            else:
                c2 = cv.compare(evalp, O(fin), O(ref), n_boot=2000); s += f'; 현재 CNN 대비 {c2["diff"]:+.4f}, p {c2["p_a_better"]:.2f}'
        s += ')'
    print(s, flush=True)
