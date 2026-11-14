"""백본별 시드 반복 요약: 융합 비중 w를 바꿔 가며 (융합 + 규칙 체인) 점수의 시드 평균, 시드 평균 확률을 쓴 경우, 바뀌는 행 수.

사용: python experiments/img/eval_arch_seeds.py   (반복 0의 5-fold OOF, ~/jaehyeok/cache/img/cnn_{tag}.npz)
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
pos = pool.index.get_indexer(evalp.index); fold_of = np.full(len(pool), -1)
for f in cv.make_folds(pool, n_repeats=1):
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
fo = fold_of[pos]
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'; cols = [f'p{k}_r0' for k in range(1, 7)]
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index)[cols].to_numpy(float)
B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)[cols].to_numpy(float)
O = lambda p: pd.DataFrame({'pred_r0': np.asarray(p).astype(int)}, index=evalp.index)  # noqa: E731


def load(tag):
    with np.load(Path.home() / 'jaehyeok' / 'cache' / 'img' / f'cnn_{tag}.npz') as z:
        C = np.zeros((len(evalp), 6))
        for k in range(5):
            m = fo == k; C[m] = z[f'out{k}_eval'][pos[m]]
    return C


G = {'resnet18 128 (현재)': ['pairdiff', 'rn18_s200', 'rn18_s300'], 'ViT-Tiny 128': ['vit_t128', 'vit_t128_s200', 'vit_t128_s300'], 'ViT-Small 224': ['vit_s224', 'vit_s224_s200', 'vit_s224_s300'],
     'ViT-Tiny 224': ['vit_t224'], 'ViT-Small 128': ['vit_s128']}
Cs = {g: [load(t) for t in tags] for g, tags in G.items()}
fin = lambda C, w: M.apply_rules(M.to_label(M.fuse(A, C, w)), B, evalp)  # noqa: E731
tab = M.apply_rules(M.to_label(A), B, evalp); base = cv.macro_f1(y, tab); WS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
print(f'이미지 없음: {base:.4f}\n시드 평균 점수 − 이미지 없음 (융합 비중 w별), 괄호는 시드별 최소~최대')
res = {}
for g, cl in Cs.items():
    s = f'{g:20s}'
    for w in WS:
        v = [cv.macro_f1(y, fin(C, w)) - base for C in cl]; res[g, w] = v; s += f' | w={w}: {np.mean(v):+.4f} ({min(v):+.4f}~{max(v):+.4f})'
    print(s, flush=True)
print('\n시드 3개의 확률을 평균한 이미지 모델 (w=0.2 / 0.3):')
ens = {g: np.mean(cl, axis=0) for g, cl in Cs.items() if len(cl) == 3}; ens['resnet18 + ViT-Tiny 128 (각 3개)'] = (ens['resnet18 128 (현재)'] + ens['ViT-Tiny 128']) / 2
ref = fin(Cs['resnet18 128 (현재)'][0], 0.2)
for g, C in ens.items():
    s = f'{g:32s} 단독 {cv.macro_f1(y, M.to_label(C)):.4f}'
    for w in (0.2, 0.3):
        p = fin(C, w); c = cv.compare(evalp, O(p), O(ref), n_boot=2000); s += f' | w={w}: {cv.macro_f1(y, p) - base:+.4f} (현재 최종 대비 {c["diff"]:+.4f}, p {c["p_a_better"]:.2f})'
    print(s, flush=True)
print('\n현재 최종(resnet18 시드 100, w=0.2) 대비 바뀌는 행 (w=0.2):')
for g in ('resnet18 128 (현재)', 'ViT-Tiny 128', 'ViT-Small 224'):
    for i, C in enumerate(Cs[g]):
        if g.startswith('resnet') and i == 0:
            continue
        p = fin(C, 0.2); ch = p != ref; fx = ch & (p == y); br = ch & (ref == y)
        print(f'  {g} 시드 {i + 1}: 바뀐 행 {int(ch.sum())} (정답으로 {int(fx.sum())}, 오답으로 {int(br.sum())}, 순이득 {int(fx.sum()) - int(br.sum())}행) | 바뀐 실험 {len(np.unique(e[ch]))}개 | 단계별 순이득 {[int((fx & (y == k)).sum()) - int((br & (y == k)).sum()) for k in range(1, 7)]}')
