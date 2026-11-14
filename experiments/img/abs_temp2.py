"""절대 온도로 변환한 열화상 신호를 '센서와 같은 방식'으로 피처화하면 도움이 되는가 (abs_temp.py의 후속).

변환: T = atmp_t + s × (thermal_t − atmp_t), s = 팔레트 역변환 값(0~1). 프레임마다 신호 9개:
  화면 평균·중앙부·90분위 온도, 온도 표준편차, 50/80/100/150/200℃ 이상 면적 비율.
피처화 (그 입력에 보이는 현재·과거 행만): 값, 30/120/300초 평균·최댓값, 15~300초 전 대비 변화, 누적 최댓값과 그 이후 하강, 임계 도달 후 경과 초 → 132개.
(1) 본체 모델에 추가  (2) 2·3단계 재판정 모델에 추가  (3) 둘 다. 공통 검증 3회 + 규칙 체인(v9). 전환별·구성별 오류 행도 본다.
사용: python experiments/img/abs_temp2.py
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
SIG = ['T_mean', 'T_center', 'T_p90', 'T_std', 'A50', 'A80', 'A100', 'A150', 'A200']


def frame_signals():
    f = C / 'img' / 'abs_temp2_signals.pkl'
    if f.exists():
        return pd.read_pickle(f)
    PT = np.load(C / 'img' / 'pool_pt128.npy', mmap_mode='r'); at = pool.atmp_t.to_numpy(float); d = np.clip(pool.thermal_t.to_numpy(float) - at, 0, None); n = len(pool)
    o = {k: np.zeros(n) for k in SIG}
    for i in range(0, n, 2000):
        s = np.asarray(PT[i:i + 2000], dtype=np.float32) / 255.0; j = slice(i, i + len(s)); fl = s.reshape(len(s), -1); dj, aj = d[j], at[j]
        o['T_mean'][j] = aj + fl.mean(1) * dj; o['T_center'][j] = aj + s[:, 32:96, 32:96].reshape(len(s), -1).mean(1) * dj
        o['T_p90'][j] = aj + np.percentile(fl, 90, axis=1) * dj; o['T_std'][j] = fl.std(1) * dj
        for th in (50, 80, 100, 150, 200):
            need = (th - aj) / np.maximum(dj, 1e-6); o[f'A{th}'][j] = (fl >= need[:, None]).mean(1) * (dj > 0)
    df = pd.DataFrame(o, index=pool.index); df.to_pickle(f)
    return df


F = frame_signals()


def family(frame):
    codes, t_all, order = M._sort_order(frame); g, t = codes[order], t_all[order]
    X = F.loc[frame.index, SIG].to_numpy(float)[order]; out = {}
    df = pd.DataFrame(X, columns=SIG); df.index = pd.to_timedelta(t, unit='s'); grp = df.groupby(g, sort=False)
    for k, c in enumerate(SIG):
        out[f'ab_{c}'] = X[:, k]
    for w in (30, 120, 300):
        r = grp.rolling(f'{w}s', min_periods=1)
        for st in ('mean', 'max'):
            v = getattr(r, st)().to_numpy()
            for k, c in enumerate(SIG):
                out[f'ab_{c}_{st}{w}'] = v[:, k]
    for w in (15, 30, 60, 120, 300):
        j = M._lag_index(g, t, w); ok = j >= 0; dd = np.where(ok[:, None], X - X[np.clip(j, 0, None)], np.nan)
        for k, c in enumerate(SIG):
            out[f'ab_{c}_d{w}'] = dd[:, k]
    cm = pd.DataFrame(X).groupby(g).cummax().to_numpy()
    for k, c in enumerate(SIG):
        out[f'ab_{c}_cmax'] = cm[:, k]; out[f'ab_{c}_drop'] = cm[:, k] - X[:, k]
    for c, ths in (('T_p90', (60, 100, 150, 200)), ('A100', (0.05,)), ('A150', (0.05,))):
        x = X[:, SIG.index(c)]
        for th in ths:
            out[f'ab_since_{c}_{th}'] = M._since(g, t, x >= th)
    res = pd.DataFrame(out); inv = np.empty(len(frame), int); inv[order] = np.arange(len(frame))
    res = res.iloc[inv]; res.index = frame.index
    return res


color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color); Ap, Ae = family(pool), family(evalp)
y = pool.label.to_numpy(int); ye = evalp.label.to_numpy(int); e = evalp.experiment_id.to_numpy()
print(f'절대 온도 피처 {Ap.shape[1]}개', flush=True)
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
Ar = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); Br = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
pa = [Ar[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]; pbs = [Br[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
CLS = {1: '원통형', 2: '파우치', 3: '각형'}; TM = {1: '가열', 2: '과충전'}
cfg = (evalp.bat_class.map(CLS) + ' ' + evalp.bat_cc.map(lambda v: f'{v:g}Ah') + ' ' + evalp.tr_method.map(TM)).to_numpy()
TESTC = ['각형 280Ah 과충전', '각형 28Ah 가열', '원통형 4.8Ah 가열', '원통형 6Ah 가열', '파우치 39Ah 가열', '파우치 73Ah 과충전']; TEST = np.isin(cfg, TESTC)
out = C / 'abs_temp2_oof.npz'
if out.exists():
    z = np.load(out); PA2, PB0, PB2 = z['PA2'], z['PB0'], z['PB2']
else:
    PA2, PB0, PB2 = (np.zeros((3, len(evalp), 6)) for _ in range(3)); t0 = time.time()
    for f in cv.make_folds(pool, n_repeats=3):
        r = f['rep']; tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        fit = lambda Xa, Xb: lgb.LGBMClassifier(**{**M.LGB_PARAMS, 'n_jobs': NJ}).fit(Xa, y[tr]).predict_proba(Xb)  # noqa: E731
        PA2[r, va] = fit(pd.concat([Xp[tr], Ap[tr]], axis=1), pd.concat([Xe[va], Ae[va]], axis=1))
        er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
        Qtr = M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er)); Qva = M.pressure_features(evalp[va], tab)
        PB0[r, va] = fit(pd.concat([Xp[tr], Qtr], axis=1), pd.concat([Xe[va], Qva], axis=1))
        PB2[r, va] = fit(pd.concat([Xp[tr], Qtr, Ap[tr]], axis=1), pd.concat([Xe[va], Qva, Ae[va]], axis=1))
        print(f"fold {r}-{f['fold']} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, PA2=PA2, PB0=PB0, PB2=PB2)


def chain(A, B):
    return pd.DataFrame({f'pred_r{r}': M.apply_rules(np.asarray(A[r]).argmax(1) + 1, B[r], evalp) for r in range(3)}, index=evalp.index)


def mf(d, m=None):
    return float(np.mean([f1_score(ye if m is None else ye[m], d[k].to_numpy() if m is None else d[k].to_numpy()[m], labels=range(1, 7), average='macro') for k in d.columns]))


def errs(d, m):
    return [round(float(np.mean([(((d[k].to_numpy() <= b) != (ye <= b)) & m).sum() for k in d.columns])), 1) for b in range(1, 6)]


cur = chain(pa, pbs)
print(f'재현 확인: 다시 만든 재판정 모델과 저장된 것의 2·3 판정 일치율 {np.mean([((PB0[r][:, 2] > PB0[r][:, 1]) == (pbs[r][:, 2] > pbs[r][:, 1])).mean() for r in range(3)]):.4f}')
print(f'현재: {mf(cur):.4f} | Test 구성 {mf(cur, TEST):.4f} | 전환별 오류 행 [1→2..5→6] 전체 {errs(cur, np.ones(len(ye), bool))} / Test 구성 {errs(cur, TEST)}')
cands = {'(1) 본체 모델에 추가': chain(PA2, pbs), '(2) 재판정 모델에 추가': chain(pa, PB2), '(3) 둘 다': chain(PA2, PB2)}
for name, d in cands.items():
    c = cv.compare(evalp, d, cur, n_boot=2000)
    print(f"{name}: {mf(d):.4f} ({c['diff']:+.4f}, 개선 확률 {c['p_a_better']:.2f}) | Test 구성 {mf(d, TEST):.4f} ({mf(d, TEST) - mf(cur, TEST):+.4f}) | 전환별 오류 행 전체 {errs(d, np.ones(len(ye), bool))} / Test 구성 {errs(d, TEST)}", flush=True)
print('\n구성별 전환 오류 행 (현재 → (3) 둘 다), 3회 평균')
rows = []
for cname in sorted(set(cfg)):
    m = cfg == cname; a, b = errs(cur, m), errs(cands['(3) 둘 다'], m)
    rows.append({'구성': cname, 'Test': cname in TESTC, '실험': len(np.unique(e[m])), **{f'{k + 1}→{k + 2}': f'{a[k]:.0f}→{b[k]:.0f}' for k in range(1, 5)}, 'F1': f"{mf(cur, m):.3f}→{mf(cands['(3) 둘 다'], m):.3f}"})
print(pd.DataFrame(rows).to_string(index=False))
