"""팀원(soomin) 변경안(img10) 채점: 현재 피처(296) vs img10, 같은 규칙 체인, 공통 검증 3회 + 이미지 융합(rep 0)."""
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
z = np.load(C / 'robust_img10_oof.npz', allow_pickle=True); assert (z['index'] == evalp.index.to_numpy()).all()
cur = ([A[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(float) for r in range(3)], [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy(float) for r in range(3)])
new = ([z['PA'][r].astype(float) for r in range(3)], [z['PB'][r].astype(float) for r in range(3)])
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2']); origin = evalp.origin.to_numpy(); e = evalp.experiment_id.to_numpy()


def v7(a, pb):
    return M.hold_stage(M.floor_stage4(M.gate_stage5(M.gate_stage6(M.redecide_23(a, pb), evalp), evalp), evalp), evalp)


def frame(P, chain):
    return pd.DataFrame({f'pred_r{r}': chain(M.to_label(P[0][r]), P[1][r]) for r in range(3)}, index=evalp.index)


mf = lambda m, g: np.mean([f1_score(y[m], g[f'pred_r{r}'].to_numpy()[m], labels=range(1, 7), average='macro') for r in range(3)])  # noqa: E731
for cname, chain in (('규칙 없음(argmax)', lambda a, pb: a), ('v7 규칙', v7), ('v8 규칙', lambda a, pb: M.apply_rules(a, pb, evalp))):
    g0, g1 = frame(cur, chain), frame(new, chain)
    r0, r1 = cv.evaluate(evalp, g0, n_boot=200), cv.evaluate(evalp, g1, n_boot=200); c = cv.compare(evalp, g1, g0, n_boot=4000)
    print(f"[{cname}] 현재 피처 {r0['cv']:.4f} {np.round(r0['per_class'], 3).tolist()}")
    print(f"{'':10s} img10    {r1['cv']:.4f} {np.round(r1['per_class'], 3).tolist()} | 차이 {c['diff']:+.4f} [{c['ci90'][0]:+.4f},{c['ci90'][1]:+.4f}] 개선 확률 {c['p_a_better']:.2f}")
    print(f"{'':10s} 반복별 차이 {[round(float(f1_score(y, g1[f'pred_r{r}'], labels=range(1, 7), average='macro') - f1_score(y, g0[f'pred_r{r}'], labels=range(1, 7), average='macro')), 4) for r in range(3)]}"
          f" | Test 구성만 {mf(TEST, g1) - mf(TEST, g0):+.4f} | 학습 출처 {mf(origin == 'train', g1) - mf(origin == 'train', g0):+.4f} / 검증 출처 {mf(origin == 'validation', g1) - mf(origin == 'validation', g0):+.4f}", flush=True)
g0, g1 = frame(cur, v7), frame(new, v7)
# 실험 절반 300회
rng = np.random.default_rng(0); ex = np.unique(e); hh = []
for _ in range(300):
    h = np.isin(e, rng.choice(ex, len(ex) // 2, replace=False)); hh.append(mf(h, g1) - mf(h, g0))
hh = np.array(hh); print(f"\n[v7 규칙] 실험 절반 300회: 차이 평균 {hh.mean():+.4f}, 음수 비율 {(hh < 0).mean():.0%}, 5~95% {np.quantile(hh, .05):+.4f}~{np.quantile(hh, .95):+.4f}")
# 구성별 오류 행
d = pd.DataFrame({'cfg': cfg, '현재': np.mean([(g0[f'pred_r{r}'].to_numpy() != y) for r in range(3)], 0), 'img10': np.mean([(g1[f'pred_r{r}'].to_numpy() != y) for r in range(3)], 0)})
t = d.groupby('cfg').sum().round(0).astype(int); t['차이'] = t['img10'] - t['현재']; print('\n구성별 오류 행 (3회 평균):'); print(t.T.to_string())
# 실험별 순이득 분포
net = pd.Series(np.sum([((g1[f'pred_r{r}'].to_numpy() == y).astype(int) - (g0[f'pred_r{r}'].to_numpy() == y).astype(int)) for r in range(3)], 0)).groupby(e).sum() / 3
print(f"\n실험별 순이득(행, 3회 평균): 좋아진 실험 {int((net > 0.5).sum())}, 나빠진 실험 {int((net < -0.5).sum())}, 그대로 {int((net.abs() <= 0.5).sum())} | 상위 5 {np.round(np.sort(net.values)[-5:], 1).tolist()} | 하위 5 {np.round(np.sort(net.values)[:5], 1).tolist()}")
# 이미지 융합 (rep 0)
pos = pool.index.get_indexer(evalp.index); fold_of = np.full(len(pool), -1)
for f in cv.make_folds(pool, n_repeats=1):
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
with np.load(C / 'img' / 'cnn_pairdiff.npz') as zz:
    Cn = np.zeros((len(evalp), 6))
    for k in range(5):
        m = fold_of[pos] == k; Cn[m] = zz[f'out{k}_eval'][pos[m]]
print('\n이미지 융합 포함 (rep 0, v7 규칙):')
O = lambda p: pd.DataFrame({'pred_r0': p}, index=evalp.index)  # noqa: E731
ref = O(v7(M.to_label(M.fuse(cur[0][0], Cn, 0.2)), cur[1][0]))
for nm, P in (('현재 피처', cur), ('img10', new)):
    for w in (0.0, 0.1, 0.2):
        p = v7(M.to_label(M.fuse(P[0][0], Cn, w)) if w > 0 else M.to_label(P[0][0]), P[1][0]); r = cv.evaluate(evalp, O(p), n_boot=50); c = cv.compare(evalp, O(p), ref, n_boot=2000)
        print(f"  {nm:8s} 융합 w={w}: {r['cv']:.4f} | 현재 최종(현재 피처, w=0.2) 대비 {c['diff']:+.4f} 개선 확률 {c['p_a_better']:.2f}")
