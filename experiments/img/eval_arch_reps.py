"""이미지 백본 비교를 공통 검증 3회(fold 나누기 3가지) 전체로: resnet18(현재) vs ViT-Tiny 128px. 융합 w + 규칙 체인(v9) 적용 후 짝지어 비교.

OOF: ~/jaehyeok/cache/img/cnn_{tag}.npz — 반복 r의 fold로 만든 것 (finetune_arch.py --rep r [--seed0 s]).
사용: python experiments/img/eval_arch_reps.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool); y = evalp.label.to_numpy(int); e = evalp.experiment_id.to_numpy()
pos = pool.index.get_indexer(evalp.index); fo = np.full((3, len(evalp)), -1)
for f in cv.make_folds(pool, n_repeats=3):
    fo[f['rep'], evalp.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
Ar = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); Br = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
A = [Ar[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(float) for r in range(3)]; B = [Br[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(float) for r in range(3)]
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])


def load(tag, r):
    with np.load(Path.home() / 'jaehyeok' / 'cache' / 'img' / f'cnn_{tag}.npz') as z:
        C = np.zeros((len(evalp), 6))
        for k in range(5):
            m = fo[r] == k; C[m] = z[f'out{k}_eval'][pos[m]]
    return C


def final(tags, w=0.2):
    return pd.DataFrame({f'pred_r{r}': M.apply_rules(M.to_label(M.fuse(A[r], load(tags[r], r), w)), B[r], evalp) for r in range(3)}, index=evalp.index)


def mf(d, m=None):
    return float(np.mean([cv.macro_f1(y if m is None else y[m], d[k].to_numpy() if m is None else d[k].to_numpy()[m]) for k in d.columns]))


SETS = {'resnet18 시드 100 (현재)': ['pairdiff', 'rn18_r1', 'rn18_r2'], 'resnet18 시드 200': ['rn18_s200', 'rn18_r1_s200', 'rn18_r2_s200'],
        'ViT-Tiny 시드 100': ['vit_t128', 'vit_t128_r1', 'vit_t128_r2'], 'ViT-Tiny 시드 200': ['vit_t128_s200', 'vit_t128_r1_s200', 'vit_t128_r2_s200']}
tab = pd.DataFrame({f'pred_r{r}': M.apply_rules(M.to_label(A[r]), B[r], evalp) for r in range(3)}, index=evalp.index)
print(f'이미지 없음 (테이블 + 규칙 체인, 3회): {mf(tab):.4f} | Test 구성 {mf(tab, TEST):.4f}')
avail = {n: t for n, t in SETS.items() if all((Path.home() / 'jaehyeok' / 'cache' / 'img' / f'cnn_{x}.npz').exists() for x in t)}
for w in (0.2, 0.3):
    print(f'\n--- 융합 비중 w={w} ---'); F = {n: final(t, w) for n, t in avail.items()}; ref = F['resnet18 시드 100 (현재)']
    for n, d in F.items():
        c0 = cv.compare(evalp, d, tab, n_boot=2000); s = f'{n:24s} {mf(d):.4f} (이미지 없음 대비 {c0["diff"]:+.4f}, p {c0["p_a_better"]:.2f}) | 반복별 {[round(cv.macro_f1(y, d[k].to_numpy()) - cv.macro_f1(y, tab[k].to_numpy()), 4) for k in d.columns]} | Test 구성 {mf(d, TEST):.4f} ({mf(d, TEST) - mf(tab, TEST):+.4f})'
        if d is not ref:
            c = cv.compare(evalp, d, ref, n_boot=2000); ch = sum(int((d[k] != ref[k]).sum()) for k in d.columns) / 3
            fx = sum(int(((d[k] != ref[k]) & (d[k] == y)).sum()) for k in d.columns) / 3; br = sum(int(((d[k] != ref[k]) & (ref[k] == y)).sum()) for k in d.columns) / 3
            s += f' | 현재 최종 대비 {c["diff"]:+.4f} (p {c["p_a_better"]:.2f}), 회당 바뀐 행 {ch:.0f} (정답으로 {fx:.0f} / 오답으로 {br:.0f})'
        print(s, flush=True)
    if 'ViT-Tiny 시드 200' in F and 'resnet18 시드 200' in F:
        c = cv.compare(evalp, F['ViT-Tiny 시드 200'], F['resnet18 시드 200'], n_boot=2000); print(f'ViT-Tiny 시드 200 vs resnet18 시드 200: {c["diff"]:+.4f} (p {c["p_a_better"]:.2f})')
