"""규칙 상수의 과적합 점검 (v9): 실험 절반(H1)의 라벨만 보고 상수를 다시 정한 뒤 나머지 절반(H2)에 적용.

데이터에서 정한 상수와 그 정하는 방법(고정된 절차):
  원통형 4단계 최소 길이   = H1 원통형 실험의 4단계 길이 최솟값
  각형 최고점 하강 임계    = 1.2 × (H1 각형 실험의 5단계 중 최고점 대비 하강 최댓값)
  급상승 판정 속도         = 1.05 × (H1 각형 실험의 4단계 중, 전압 급락 이후 직전 관측 대비 초당 상승 최댓값)
  급상승 10초 상승         = 1.25 × (H1 각형 실험의 4단계 중, 전압 급락 이후 10초 상승 최댓값)
가이드에서 온 값(0.25, 15초)과 규칙 구조는 그대로 둔다. 모델 예측은 공통 검증 OOF(3회)를 쓴다.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(sys.argv[1]).resolve(); sys.path[:0] = [str(REPO), str(REPO / 'experiments' / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool); y = evalp.label.to_numpy(int); e = evalp.experiment_id.to_numpy()
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
a = [A[f'pred_r{r}'].to_numpy(int) for r in range(3)]; pb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
# 실험별 통계 (원본 5초 행, 라벨 사용 = 설계 단계에서만)
st = {}
for ex, g in pool.sort_values('relative_time_s').groupby('experiment_id'):
    t = g.relative_time_s.to_numpy(float); lab = g.label.to_numpy(int); t1 = g.meas_t1.to_numpy(float); v = g.meas_v.to_numpy(float); S = t1 + g.meas_t2.to_numpy(float)
    bc, cc, tm = int(g.bat_class.iloc[0]), float(g.bat_cc.iloc[0]), int(g.tr_method.iloc[0]); d = {'cyl': bc == 1, 'pris': (bc, cc, tm) in ((3, 28.0, 1), (3, 280.0, 2))}
    if (lab == 4).any() and (lab == 5).any():
        i4, i5 = int(np.argmax(lab == 4)), int(np.argmax(lab == 5)); d['dur4'] = t[i5] - t[i4]
        dv = np.r_[0, np.diff(v)]; ev = (dv <= -0.56) | (v < 2.5)
        if ev.any():
            ivd = int(np.argmax(ev)); rate = np.r_[0, np.diff(t1) / np.diff(t)]; rate[:ivd + 1] = 0
            j = np.searchsorted(t, t - 10, side='right') - 1; r10 = np.where(j >= ivd, t1 - t1[np.clip(j, 0, None)], 0.0)
            m4 = (np.arange(len(t)) < i5) & (np.arange(len(t)) > ivd)
            d['rate4'] = float(rate[m4].max()) if m4.any() else 0.0; d['r10_4'] = float(r10[m4].max()) if m4.any() else 0.0
    if (lab == 5).any():
        m5 = lab == 5; d['dip5'] = float((np.maximum.accumulate(S) - S)[m5].max())
    st[ex] = d
exps = np.array(sorted(st)); FIX = dict(STAGE4_MIN_S=M.STAGE4_MIN_S, PEAK_DROP=M.PEAK_DROP, T1_RATE_STAGE5=M.T1_RATE_STAGE5, SURGE_RISE=M.SURGE_RISE)


def derive(h1):
    g = lambda key, cond: [st[x][key] for x in h1 if st[x].get(cond) and key in st[x]]  # noqa: E731
    return dict(STAGE4_MIN_S=float(min(g('dur4', 'cyl'))), PEAK_DROP=1.2 * max(g('dip5', 'pris')), T1_RATE_STAGE5=1.05 * max(g('rate4', 'pris')), SURGE_RISE=1.25 * max(g('r10_4', 'pris')))


def chain(const):
    old = {k: getattr(M, k) for k in const}
    for k, val in const.items():
        setattr(M, k, val)
    try:
        return [M.apply_rules(a[r], pb[r], evalp) for r in range(3)]
    finally:
        for k, val in old.items():
            setattr(M, k, val)


mf = lambda m, P: np.mean([f1_score(y[m], p[m], labels=range(1, 7), average='macro') for p in P])  # noqa: E731
none = a; fixed = chain(FIX)
rng = np.random.default_rng(0); out = []; consts = []
for it in range(40):
    h1 = rng.choice(exps, len(exps) // 2, replace=False); m2 = ~np.isin(e, h1)
    c = derive(h1); consts.append(c); der = chain(c)
    out.append((mf(m2, none), mf(m2, fixed), mf(m2, der)))
o = np.array(out); cd = pd.DataFrame(consts)
print('최종 상수:', {k: round(v, 3) for k, v in FIX.items()})
print('절반에서 다시 정한 상수 (40회): ' + ' | '.join(f"{k} 중앙 {cd[k].median():.2f} (범위 {cd[k].min():.2f}~{cd[k].max():.2f})" for k in cd.columns))
print(f"나머지 절반에서의 Macro F1 (40회 평균): 규칙 없음 {o[:, 0].mean():.4f} | 최종 상수 {o[:, 1].mean():.4f} (+{(o[:, 1] - o[:, 0]).mean():.4f}) | 절반에서 정한 상수 {o[:, 2].mean():.4f} (+{(o[:, 2] - o[:, 0]).mean():.4f})")
d = o[:, 2] - o[:, 1]; print(f"절반에서 정한 상수 − 최종 상수: 평균 {d.mean():+.4f}, 최소 {d.min():+.4f}, 최대 {d.max():+.4f} | 규칙 효과가 양수인 비율 (절반에서 정한 상수): {((o[:, 2] - o[:, 0]) > 0).mean():.0%}")
