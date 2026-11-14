"""후기 융합(late fusion): 세션 A 테이블 모델의 OOF 확률 × CNN OOF 확률 (재학습 없음, 같은 fold).

P = P_tab^(1-w) · P_cnn^w (정규화) → argmax.  w=0이면 테이블 모델 그대로.
w는 이 OOF에서 고르므로 약간 낙관적(자유도 1). 이미지 바꾸기: CNN 확률을 같은 fold의 다른 행 것으로 교체.
예) python latefuse.py tab_all_pool single temporal temporal_pt
"""
import sys
import numpy as np
import pandas as pd
import imglib as L
from imglib import cv

tab_name, tags = sys.argv[1], sys.argv[2:]
pool, evalp = L.load_all()
N = len(pool)
pos = pool.index.get_indexer(evalp.index)
folds = cv.make_folds(pool, n_repeats=1)
fold_of = np.full(N, -1)
for f in folds:
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
T = pd.read_pickle(L.HERE.parent / f'results/oof/{tab_name}.pkl').reindex(evalp.index)
Pt = T[[f'p{k}_r0' for k in range(1, 7)]].to_numpy(float)
base = T[['pred_r0']].astype(int)
yy = evalp.label.to_numpy()
ON = L.onset_rows(evalp)
onacc = lambda o: [round(float((o.pred_r0.to_numpy()[r] == yy[r]).mean()), 3) for r in ON.values()]
print(cv.summary_line(tab_name + ' (r0)', cv.evaluate(evalp, base)), '| 시작 행 정확도 4/5/6', onacc(base), flush=True)


def oof_of(P):
    return pd.DataFrame({'pred_r0': P.argmax(1) + 1}, index=evalp.index)


def cnn(tag):
    with np.load(L.ICACHE / f'cnn_{tag}.npz') as z:
        P = np.zeros((len(evalp), 6))
        for k in range(5):
            m = fold_of[pos] == k
            P[m] = z[f'out{k}_eval'][pos[m]]
    return P


C = {t: cnn(t) for t in tags}
if len(tags) > 1:
    C['avg(' + '+'.join(tags) + ')'] = np.mean([C[t] for t in tags], 0)
rng = np.random.default_rng(0)
for tag, Pc in C.items():
    print(f'--- CNN {tag}: 이미지 단독 {cv.macro_f1(yy, Pc.argmax(1) + 1):.4f}')
    perm = np.arange(len(evalp))
    for k in range(5):
        m = np.where(fold_of[pos] == k)[0]
        perm[m] = rng.permutation(m)
    for w in (0.05, 0.1, 0.2, 0.3, 0.5):
        P = np.log(Pt + 1e-9) * (1 - w) + np.log(Pc + 1e-6) * w
        o = oof_of(P)
        r = cv.evaluate(evalp, o, n_boot=200)
        c = cv.compare(evalp, o, base, n_boot=2000)
        o2 = oof_of(np.log(Pt + 1e-9) * (1 - w) + np.log(Pc[perm] + 1e-6) * w)
        print(f"  w={w:.2f} CV {r['cv']:.4f} long {r['long_exps']:.3f} short {r['short_exps']:.3f} per-class {np.round(r['per_class'], 3).tolist()} | "
              f"vs tab diff {c['diff']:+.4f} ci90 [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p_better {c['p_a_better']:.2f} | "
              f"2→3 {((yy == 2) & (o.pred_r0 == 3)).sum()} 3→2 {((yy == 3) & (o.pred_r0 == 2)).sum()} | "
              f"시작 행 4/5/6 {onacc(o)} | 이미지 바꾸기: 변경 {(o2.pred_r0 != o.pred_r0).mean():.3f}, 바꾼 뒤 F1 {cv.macro_f1(yy, o2.pred_r0):.4f}", flush=True)
        if w == 0.2:
            cv.save_result(f'img_latefuse_{tab_name}_{tag.split("(")[0]}_w20', cv.evaluate(evalp, o), o)
print(f"(참고) 테이블 2→3 {((yy == 2) & (base.pred_r0 == 3)).sum()} 3→2 {((yy == 3) & (base.pred_r0 == 2)).sum()}")

# --- w를 fold 밖에서 고르는 중첩 평가 (낙관 편향 제거): fold k의 w = 나머지 4개 fold OOF에서 Macro F1 최대인 값 ---
WS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
fo = fold_of[pos]
for tag, Pc in C.items():
    lt, lc = np.log(Pt + 1e-9), np.log(Pc + 1e-6)
    pred = np.zeros(len(evalp), int); chosen = []
    for k in range(5):
        tr, va = fo != k, fo == k
        best = max(WS, key=lambda w: cv.macro_f1(yy[tr], (lt[tr] * (1 - w) + lc[tr] * w).argmax(1) + 1))
        chosen.append(best)
        pred[va] = (lt[va] * (1 - best) + lc[va] * best).argmax(1) + 1
    o = pd.DataFrame({'pred_r0': pred}, index=evalp.index)
    c = cv.compare(evalp, o, base)
    r = cv.evaluate(evalp, o)
    cv.save_result(f'img_latefuse_{tab_name}_{tag.split("(")[0]}_nestedw', r, o)
    print(f"[중첩 w] CNN {tag}: fold별 w {chosen} → CV {r['cv']:.4f} [{r['boot_ci90'][0]:.3f},{r['boot_ci90'][1]:.3f}] long {r['long_exps']:.3f} short {r['short_exps']:.3f} "
          f"per-class {np.round(r['per_class'], 3).tolist()} | 시작 행 4/5/6 {onacc(o)} | vs tab diff {c['diff']:+.4f} ci90 [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p_better {c['p_a_better']:.2f}", flush=True)
