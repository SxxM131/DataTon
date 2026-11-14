"""Macro F1 기준 판정값 조정: (1) 2·3단계 재판정 기준 τ (p3/(p2+p3) > τ 이면 3), (2) 4·5·6단계 확률 가중 w4·w5·w6.

argmax는 정확도를 최대화하지만 Macro F1은 행이 적은 단계의 오류를 더 크게 본다 → 판정 기준을 옮기면 이득일 수 있다.
정직한 평가: 각 fold의 값은 같은 반복의 나머지 4개 fold OOF에서만 고른다 (그 fold의 라벨은 쓰지 않음).
기준: 현재 최종 체인(apply_rules)의 테이블 부분, 공통 검증 3회.
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

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
R = HERE.parent / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
y = evalp.label.to_numpy(int); e = evalp.experiment_id.to_numpy(); t = evalp.relative_time_s.to_numpy(float)
PA = [A[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(float) for r in range(3)]
PB = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(float) for r in range(3)]
# 규칙에 쓰는 근거 신호는 한 번만 계산
climb = M.still_climbing(evalp); t_surge = M._cfg_mask(evalp, M.SURGE_CFGS); t_vdd = M._cfg_mask(evalp, M.VDD_CFGS)
no_ev5 = (t_surge & ~M.surge_seen(evalp)) | (t_vdd & ~M.second_drop_seen(evalp)); vd = M.vdrop_seen(evalp)
order = np.lexsort((evalp.sample_id.astype(str).to_numpy(), t, pd.factorize(e, sort=True)[0]))
bounds = np.flatnonzero(np.r_[True, e[order][1:] != e[order][:-1], True])
groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def chain(r, tau=0.5, w=(1.0, 1.0, 1.0)):
    pa = PA[r] * np.array([1, 1, 1, w[0], w[1], w[2]])
    x = pa.argmax(1) + 1
    s = (x == 2) | (x == 3)
    q = PB[r][:, 2] / (PB[r][:, 1] + PB[r][:, 2] + 1e-12)
    x[s] = np.where(q[s] > tau, 3, 2)
    x[(x == 6) & climb] = 5
    x[(x >= 5) & no_ev5] = 4
    for ix in groups:
        z = x[ix]
        if t_vdd[ix[0]] and (z >= 4).any():
            t_in = t[ix][int(np.argmax(z >= 4))]; z = np.where((z > 4) & (t[ix] - t_in < M.STAGE4_MIN_S - 1e-9), 4, z)
        z = np.where(np.maximum.accumulate(z >= 4) & vd[ix] & (z < 4), 4, z)
        z = np.where(np.maximum.accumulate(z >= 5) & (z < 5), 5, z)
        x[ix] = z
    return x


for r in range(3):
    assert (chain(r) == M.apply_rules(M.to_label(PA[r]), PB[r], evalp)).all(), '빠른 체인이 model.apply_rules와 다름'
print('빠른 체인 = model.apply_rules 확인', flush=True)
mf = lambda m, p: f1_score(y[m], p[m], labels=range(1, 7), average='macro')  # noqa: E731
TAUS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8); WS = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6, 2.0)


def tune(r, m, do_tau=True, do_w=True):
    tau, w = 0.5, [1.0, 1.0, 1.0]
    best = mf(m, chain(r, tau, w))
    for _ in range(2):
        if do_tau:
            for c in TAUS:
                s = mf(m, chain(r, c, w))
                if s > best + 1e-9:
                    best, tau = s, c
        if do_w:
            for k in range(3):
                for c in WS:
                    w2 = list(w); w2[k] = c; s = mf(m, chain(r, tau, w2))
                    if s > best + 1e-9:
                        best, w = s, w2
    return tau, tuple(w)


folds = cv.make_folds(pool)
base = pd.DataFrame({f'pred_r{r}': chain(r) for r in range(3)}, index=evalp.index)
for name, do_tau, do_w in (('τ만', True, False), ('가중만', False, True), ('τ + 가중', True, True)):
    out = base.copy(); chosen = []
    for f in folds:
        r = f['rep']; va = np.isin(e, f['valid_exps'])
        tau, w = tune(r, ~va, do_tau, do_w); chosen.append((tau, *w))
        out.loc[va, f'pred_r{r}'] = chain(r, tau, w)[va]
    c = cv.compare(evalp, out, base, n_boot=3000); rr = cv.evaluate(evalp, out, n_boot=100)
    ch = pd.DataFrame(chosen, columns=['tau', 'w4', 'w5', 'w6'])
    print(f"[fold 밖에서 고름] {name:8s} CV {rr['cv']:.4f} ({c['diff']:+.4f} [{c['ci90'][0]:+.4f},{c['ci90'][1]:+.4f}], p {c['p_a_better']:.2f}) {np.round(rr['per_class'], 3).tolist()}")
    print('    고른 값의 분포:', {k: ch[k].value_counts().sort_index().to_dict() for k in ch.columns}, flush=True)
# 참고: 전체 OOF에서 고른 값 (낙관적)
allm = np.ones(len(y), bool)
for r in range(3):
    tau, w = tune(r, allm); print(f"[참고, 전체에서 고름] rep {r}: τ={tau}, w={w}, {mf(allm, chain(r)):.4f} → {mf(allm, chain(r, tau, w)):.4f}")
