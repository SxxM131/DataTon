"""견고성 검증: 각 단계(2·3·4·5) 구간을 복제해 3배로 늘렸을 때 '시간이 지났다는 이유만으로' 다음 단계로 넘기는지. 구성별.

늘린 행의 라벨은 원래 단계 그대로 (다음 단계의 이벤트가 아직 없으므로). 공통 검증 rep 0, 모델은 원래 학습 fold로 학습.
"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
y = pool.label.to_numpy(int)


def stretch(g, s, K):
    g = g.sort_values('relative_time_s').reset_index(drop=True)
    lab = g.label.to_numpy(int)
    if not (lab == s).any() or not (lab > s).any():
        return None
    i0 = int(np.argmax(lab == s)); i1 = int(np.argmax(lab > s))
    a, b = i0 + 2, i1 - 1
    if b - a < 3:
        return None
    dt = float(np.median(np.diff(g.relative_time_s)))
    blk = g.iloc[a:b]; parts = [g.iloc[:b]]; shift = 0.0
    for k in range(K):
        c = blk.copy(); shift += (blk.relative_time_s.iloc[-1] - blk.relative_time_s.iloc[0]) + dt
        c['relative_time_s'] = c.relative_time_s + shift; c['sample_id'] = c.sample_id.astype(str) + f'#{k}'; parts.append(c)
    tail = g.iloc[b:].copy(); tail['relative_time_s'] = tail.relative_time_s + shift
    out = pd.concat(parts + [tail], ignore_index=True)
    out['src_id'] = out.sample_id.astype(str).str.split('#').str[0]
    return out


PR = {**M.LGB_PARAMS, 'n_jobs': 6}
Xp = M.build_features(pool, color)
res = []
for f in cv.make_folds(pool, n_repeats=1):
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
    va = evalp[evalp.experiment_id.isin(f['valid_exps'])]
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    a = lgb.LGBMClassifier(**PR).fit(Xp[tr], y[tr])
    b = lgb.LGBMClassifier(**PR).fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1), y[tr])
    for s in (2, 3, 4, 5):
        for K in (0, 2):
            frames = [stretch(g, s, K) for _, g in va.groupby('experiment_id', sort=True)]
            frames = [x for x in frames if x is not None]
            if not frames:
                continue
            fr = pd.concat(frames, ignore_index=True)
            col = color.reindex(fr.src_id.values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
            X = M.build_features(fr, col)
            p = M.redecide_23(a.predict(X), b.predict_proba(pd.concat([X, M.pressure_features(fr, tab)], axis=1)))
            res.append(pd.DataFrame({'s': s, 'K': K, 'cfg': fr.bat_class.astype(int).astype(str) + '/' + fr.bat_cc.astype(str) + '/' + fr.tr_method.astype(int).astype(str),
                                     'e': fr.experiment_id, 'y': fr.label.to_numpy(int), 'p': p}))
    print('fold', f['fold'], flush=True)
d = pd.concat(res)
d['early'] = (d.y == d.s) & (d.p > d.s)                # 그 단계인데 다음 단계 이상으로 예측
d['is_s'] = d.y == d.s
t = d.groupby(['s', 'cfg', 'K']).agg(n_exp=('e', 'nunique'), rows=('is_s', 'sum'), early=('early', 'sum')).reset_index()
w = t.pivot_table(index=['s', 'cfg'], columns='K', values=['rows', 'early', 'n_exp']).astype(int)
out = pd.DataFrame({'실험': w[('n_exp', 0)], '원래 행': w[('rows', 0)], '원래 조기 예측': w[('early', 0)], '3배 행': w[('rows', 2)], '3배 조기 예측': w[('early', 2)]})
out['늘린 구간의 조기 비율'] = ((out['3배 조기 예측'] - out['원래 조기 예측']) / (out['3배 행'] - out['원래 행']).clip(lower=1)).round(2)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 100)
print(out.to_string())
d.to_pickle(C / 'robust_stretch_all.pkl')
