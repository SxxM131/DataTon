"""분석 전용 (제출 금지): 원본 구축 가이드의 판정 규칙을 '문서 그대로'(실험 전체 정규화 + 미래 창) 적용하면 CV가 어디까지 가는가.

리더보드 상위 점수(0.94)가 규정을 지키는 모델로 가능한 수준인지 가늠하기 위한 실험이다.
여기 피처는 실험 전체 min-max 정규화와 미래 관측을 쓰므로 대회 규정 위반이며, 최종 모델에 절대 쓰지 않는다.
"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'common'), str(HERE.parent / 'tab'), str(HERE.parent / 'rules')]
import cv, feats, rules  # noqa: E402

DATA = Path.home() / 'data' / 'battery_data_light'; CACHE = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(DATA); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])


def mm(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else x * 0.0


def doc_feats(fr):
    t_all = fr['relative_time_s'].to_numpy(float)
    res = None
    for e, idx in fr.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t_all[idx], kind='mergesort')]
        f = fr.iloc[idx]
        t = t_all[idx]; p = f.meas_p.to_numpy(float); v = f.meas_v.to_numpy(float); t1 = f.meas_t1.to_numpy(float); t2 = f.meas_t2.to_numpy(float)
        n = len(t)
        pn, vn, tn = mm(p), mm(v), mm(t1)
        def fwd(x, w):
            j = np.searchsorted(t, t + w, side='left'); ok = j < n
            d = np.full(n, np.nan); d[ok] = x[j[ok]] - x[ok]; return d
        special = (f.bat_class.iloc[0] == 3 and f.bat_cc.iloc[0] == 280 and f.tr_method.iloc[0] == 1) or \
                  (f.bat_class.iloc[0] == 2 and f.bat_cc.iloc[0] == 73 and f.tr_method.iloc[0] == 1 and f.soc_status.iloc[0] == 1)
        g = (t1 >= 50) & (t1 <= 90) & (np.maximum.accumulate(t1) >= 45)
        fd = fwd(pn, 20.0)
        T3 = np.nan
        for th in ([0.04, 0.03, 0.02] if special else [0.02]):
            m = (fd >= th) & g
            if m.any():
                T3 = t[m][0]; break
        dv15 = -fwd(vn, 15.0)                     # 앞으로 15초 안의 하강량 (정규화)
        m4 = dv15 >= 0.25
        T4 = t[m4][0] if m4.any() else np.nan
        T6 = t[np.argmax(t1 + t2)]
        m0 = (v <= 1e-9)
        T50 = t[m0][0] if m0.any() else np.nan
        dt10 = fwd(tn, 10.0)
        m5 = dt10 >= 0.07
        T5t = t[m5][0] if m5.any() else np.nan
        o = {'n_pn': pn, 'n_vn': vn, 'n_tn': tn, 'n_fd20': np.nan_to_num(fd), 'n_dv15': np.nan_to_num(dv15), 'n_dt10': np.nan_to_num(dt10),
             'n_frac': t / t[-1]}
        for k, T in [('T3', T3), ('T4', T4), ('T6', T6), ('T50', T50), ('T5t', T5t)]:
            o['n_d' + k] = (t - T) if np.isfinite(T) else np.full(n, -9999.0)
        if res is None:
            res = {k: np.full(len(fr), np.nan) for k in o}
        for k, a in o.items():
            res[k][idx] = a
    return pd.DataFrame(res, index=fr.index)


def F(fr, nc):
    X = pd.concat([feats.build(fr, color, {'extra': True, 'cum': False}), rules.rule_features(fr)], axis=1)
    return pd.concat([X, doc_feats(fr)], axis=1) if nc else X


P = dict(objective='multiclass', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.3, max_bin=63, reg_lambda=1, n_jobs=6, verbose=-1, random_state=42)
y = pool['label'].to_numpy(int)
base = None
for name, nc in [('규정 준수(현재 최종 후보)', False), ('문서 규칙 그대로(규정 위반)', True)]:
    Xp, Xe = F(pool, nc), F(evalp, nc)
    oof = pd.DataFrame(index=evalp.index)
    for f in cv.make_folds(pool, n_repeats=1):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        oof.loc[evalp.index[va], 'pred_r0'] = lgb.LGBMClassifier(**P).fit(Xp[tr], y[tr]).predict(Xe[va])
    oof = oof.astype(int)
    r = cv.evaluate(evalp, oof, n_boot=300)
    line = f"{name:24s} CV {r['cv']:.4f} [{r['boot_ci90'][0]:.3f},{r['boot_ci90'][1]:.3f}] | long {r['long_exps']:.3f} short {r['short_exps']:.3f} | per-class {np.round(r['per_class'], 3).tolist()} | Test 크기 모의 {r['test11_mean']:.3f}±{r['test11_sd']:.3f}"
    if base is not None:
        c = cv.compare(evalp, oof, base, n_boot=1000); line += f" | 차이 {c['diff']:+.4f} p {c['p_a_better']:.2f}"
    else:
        base = oof
    print(line, flush=True)
