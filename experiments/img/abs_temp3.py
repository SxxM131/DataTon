"""절대 온도 변환으로만 가능한 신호: 화소별 온도 변화(℃). 상대 눈금에서는 프레임마다 눈금이 달라 프레임 차이가 의미 없지만,
T = atmp_t + s × (thermal_t − atmp_t)로 바꾸면 '직전 관측 대비 화소별 몇 ℃ 올랐나'를 잴 수 있다 (분출·화염·냉각).

프레임마다 (직전 관측은 그 입력에 보이는 바로 앞 행): 화소 온도 변화의 최댓값·99분위·평균, 10/20/50℃ 넘게 오른 면적, 10/20℃ 넘게 내려간 면적 → 초당 값으로도.
피처화: 값, 30/120/300초 최댓값, 누적 최댓값, 임계 도달 후 경과 초. 본체 모델·재판정 모델에 추가해 공통 검증 3회 + 규칙 체인(v9), 전환별 오류 행.
사용: python experiments/img/abs_temp3.py
"""
import sys, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

C = Path.home() / 'jaehyeok' / 'cache'; NJ = 6
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
ids = np.load(C / 'img' / 'pool_ids.npy'); assert (ids == pool.sample_id.to_numpy().astype(str)).all()
PT = np.load(C / 'img' / 'pool_pt128.npy', mmap_mode='r')
AT = pool.atmp_t.to_numpy(np.float32); DR = np.clip(pool.thermal_t.to_numpy(np.float32) - AT, 0, None)
NAMES = ['dmax', 'dp99', 'dmean', 'up10', 'up20', 'up50', 'dn10', 'dn20']


def diff_signals(frame, tag):
    f = C / 'img' / f'abs_temp3_{tag}.pkl'
    if f.exists():
        return pd.read_pickle(f)
    pos = pool.index.get_indexer(frame.index); prev = pos[M.prev_index(frame.reset_index(drop=True))]; n = len(frame); o = {k: np.zeros(n, np.float32) for k in NAMES}
    for i in range(0, n, 1500):
        a, b = pos[i:i + 1500], prev[i:i + 1500]
        Ta = AT[a][:, None] + (np.asarray(PT[np.sort(a)], dtype=np.float32)[np.argsort(np.argsort(a))].reshape(len(a), -1) / 255.0) * DR[a][:, None]
        Tb = AT[b][:, None] + (np.asarray(PT[np.sort(b)], dtype=np.float32)[np.argsort(np.argsort(b))].reshape(len(b), -1) / 255.0) * DR[b][:, None]
        d = Ta - Tb; j = slice(i, i + len(a))
        o['dmax'][j] = d.max(1); o['dp99'][j] = np.percentile(d, 99, axis=1); o['dmean'][j] = d.mean(1)
        for th in (10, 20, 50):
            o[f'up{th}'][j] = (d >= th).mean(1)
        for th in (10, 20):
            o[f'dn{th}'][j] = (d <= -th).mean(1)
    df = pd.DataFrame(o, index=frame.index); df.to_pickle(f)
    return df


def family(frame, S):
    codes, t_all, order = M._sort_order(frame); g, t = codes[order], t_all[order]
    X = S.loc[frame.index, NAMES].to_numpy(float)[order]
    prev_t = pd.Series(t).groupby(g).shift(1).to_numpy(); dt = np.where(np.isnan(prev_t), np.nan, t - prev_t)
    out = {}
    for k, c in enumerate(NAMES):
        out[f'dd_{c}'] = X[:, k]
    for k, c in enumerate(NAMES[:3]):
        out[f'dd_{c}_ps'] = X[:, k] / dt                                             # 초당 ℃
    V = pd.DataFrame({**{c: X[:, k] for k, c in enumerate(NAMES)}, **{f'{c}_ps': X[:, k] / dt for k, c in enumerate(NAMES[:3])}}); cols = list(V.columns)
    V.index = pd.to_timedelta(t, unit='s'); grp = V.groupby(g, sort=False)
    for w in (30, 120, 300):
        v = grp.rolling(f'{w}s', min_periods=1).max().to_numpy()
        for k, c in enumerate(cols):
            out[f'dd_{c}_max{w}'] = v[:, k]
    cm = pd.DataFrame(V.to_numpy()).groupby(g).cummax().to_numpy()
    for k, c in enumerate(cols):
        out[f'dd_{c}_cmax'] = cm[:, k]
    for c, ths in (('dmax', (10, 20, 50, 100)), ('up20', (0.01, 0.05)), ('dn20', (0.01, 0.05))):
        x = X[:, NAMES.index(c)]
        for th in ths:
            out[f'dd_since_{c}_{th}'] = M._since(g, t, x >= th)
    res = pd.DataFrame(out); inv = np.empty(len(frame), int); inv[order] = np.arange(len(frame)); res = res.iloc[inv]; res.index = frame.index
    return res


t0 = time.time(); Sp, Se = diff_signals(pool, 'pool'), diff_signals(evalp, 'evalp'); Dp, De = family(pool, Sp), family(evalp, Se)
print(f'화소 온도 변화 피처 {Dp.shape[1]}개 ({time.time() - t0:.0f}s)', flush=True)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int); ye = evalp.label.to_numpy(int); e = evalp.experiment_id.to_numpy()
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
Ar = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); Br = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
pa = [Ar[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]; pbs = [Br[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
CLS = {1: '원통형', 2: '파우치', 3: '각형'}; TM = {1: '가열', 2: '과충전'}
cfg = (evalp.bat_class.map(CLS) + ' ' + evalp.bat_cc.map(lambda v: f'{v:g}Ah') + ' ' + evalp.tr_method.map(TM)).to_numpy()
TESTC = ['각형 280Ah 과충전', '각형 28Ah 가열', '원통형 4.8Ah 가열', '원통형 6Ah 가열', '파우치 39Ah 가열', '파우치 73Ah 과충전']; TEST = np.isin(cfg, TESTC)
out = C / 'abs_temp3_oof.npz'
if out.exists():
    z = np.load(out); PA3, PB3 = z['PA3'], z['PB3']
else:
    PA3, PB3 = np.zeros((3, len(evalp), 6)), np.zeros((3, len(evalp), 6)); t0 = time.time()
    for f in cv.make_folds(pool, n_repeats=3):
        r = f['rep']; tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        fit = lambda Xa, Xb: lgb.LGBMClassifier(**{**M.LGB_PARAMS, 'n_jobs': NJ}).fit(Xa, y[tr]).predict_proba(Xb)  # noqa: E731
        PA3[r, va] = fit(pd.concat([Xp[tr], Dp[tr]], axis=1), pd.concat([Xe[va], De[va]], axis=1))
        er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
        PB3[r, va] = fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er)), Dp[tr]], axis=1), pd.concat([Xe[va], M.pressure_features(evalp[va], tab), De[va]], axis=1))
        print(f"fold {r}-{f['fold']} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, PA3=PA3, PB3=PB3)
chain = lambda A, B: pd.DataFrame({f'pred_r{r}': M.apply_rules(np.asarray(A[r]).argmax(1) + 1, B[r], evalp) for r in range(3)}, index=evalp.index)  # noqa: E731
mf = lambda d, m=None: float(np.mean([f1_score(ye if m is None else ye[m], d[k].to_numpy() if m is None else d[k].to_numpy()[m], labels=range(1, 7), average='macro') for k in d.columns]))  # noqa: E731
errs = lambda d, m: [round(float(np.mean([(((d[k].to_numpy() <= b) != (ye <= b)) & m).sum() for k in d.columns])), 1) for b in range(1, 6)]  # noqa: E731
cur = chain(pa, pbs); ALL = np.ones(len(ye), bool)
print(f'현재: {mf(cur):.4f} | Test 구성 {mf(cur, TEST):.4f} | 전환별 오류 행 [1→2..5→6] 전체 {errs(cur, ALL)} / Test 구성 {errs(cur, TEST)}')
cands = {'(1) 본체 모델에 추가': chain(PA3, pbs), '(2) 재판정 모델에 추가': chain(pa, PB3), '(3) 둘 다': chain(PA3, PB3)}
for name, d in cands.items():
    c = cv.compare(evalp, d, cur, n_boot=2000)
    print(f"{name}: {mf(d):.4f} ({c['diff']:+.4f}, 개선 확률 {c['p_a_better']:.2f}) | Test 구성 {mf(d, TEST):.4f} ({mf(d, TEST) - mf(cur, TEST):+.4f}) | 전환별 오류 행 전체 {errs(d, ALL)} / Test 구성 {errs(d, TEST)}", flush=True)
print('\n구성별 전환 오류 행 (현재 → (1) 본체 모델에 추가), 3회 평균'); rows = []
for cname in sorted(set(cfg)):
    m = cfg == cname; a, b = errs(cur, m), errs(cands['(1) 본체 모델에 추가'], m)
    rows.append({'구성': cname, 'Test': cname in TESTC, **{f'{k + 1}→{k + 2}': f'{a[k]:.0f}→{b[k]:.0f}' for k in range(1, 5)}, 'F1': f"{mf(cur, m):.3f}→{mf(cands['(1) 본체 모델에 추가'], m):.3f}"})
print(pd.DataFrame(rows).to_string(index=False))
