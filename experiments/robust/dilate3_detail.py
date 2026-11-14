"""3단계가 느려진 합성 실험에서 무엇이 틀리는지: 구성별, '되돌아가지 않기' 유무별."""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402
from dilate_test import dilate, pool, evalp, color  # noqa: E402

yp = pool.label.to_numpy(int); PR = {**M.LGB_PARAMS, 'n_jobs': 6}; Xp = M.build_features(pool, color); res = []
cfgname = lambda f: f.bat_class.astype(int).astype(str) + '/' + f.bat_cc.astype(str) + '/' + f.tr_method.astype(int).astype(str)  # noqa: E731
for f in cv.make_folds(pool, n_repeats=1):
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp[evalp.experiment_id.isin(f['valid_exps'])]
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    a = lgb.LGBMClassifier(**PR).fit(Xp[tr], yp[tr]); b = lgb.LGBMClassifier(**PR).fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1), yp[tr])
    for K in (1, 2, 3):
        fr = pd.concat([x for x in (dilate(g, 3, K) for _, g in va.groupby('experiment_id', sort=True)) if x is not None], ignore_index=True)
        col = color.reindex(fr.src_id.values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
        X = M.build_features(fr, col); pa = a.predict(X); pb = b.predict_proba(pd.concat([X, M.pressure_features(fr, tab)], axis=1))
        raw = M.redecide_23(pa, pb)
        nohold = M.floor_stage4(M.gate_stage5(M.gate_stage6(raw, fr), fr), fr)
        vd = M.rule_features(fr).r_since_vdrop.to_numpy() >= 0
        res.append(pd.DataFrame({'K': K, 'cfg': cfgname(fr).to_numpy(), 'e': fr.experiment_id.to_numpy(), 'y': fr.label.to_numpy(int), 'raw': raw, 'nohold': nohold, 'full': M.hold_stage(nohold, fr), 'vd': vd}))
    print('fold', f['fold'], flush=True)
d = pd.concat(res)
d.to_pickle(Path.home() / 'jaehyeok' / 'cache' / 'robust_dilate3.pkl')
pd.set_option('display.width', 230)
for K in (1, 2, 3):
    x = d[d.K == K]; s = f'×{K}:'
    for m in ('raw', 'nohold', 'full'):
        f = f1_score(x.y, x[m], labels=range(1, 7), average=None, zero_division=0)
        s += f" | {m} Macro {f.mean():.4f} (4·5·6 F1 {f[3]:.3f}/{f[4]:.3f}/{f[5]:.3f}), 3을 4 이상으로 {int(((x.y == 3) & (x[m] >= 4)).sum())}"
    print(s)
x = d[d.K == 3]
t = x.assign(pre_raw=(x.y == 3) & (x.raw >= 4), pre_full=(x.y == 3) & (x.full >= 4), pre_vd=(x.y == 3) & x.vd, n3=x.y == 3, e45=np.isin(x.y, (4, 5)) & (x.full != x.y), e56=np.isin(x.y, (5, 6)) & (x.full != x.y) & np.isin(x.full, (5, 6))).groupby('cfg')[['n3', 'pre_raw', 'pre_full', 'pre_vd', 'e45', 'e56']].sum()
print('\n×3, 구성별: 3단계 행 / 모델이 4 이상으로 낸 행(보정 전) / 최종 / 그중 전압 급락 이벤트가 켜진 행 / 4·5 오류 / 5↔6 오류'); print(t.to_string())
g = x.assign(pre=(x.y == 3) & (x.full >= 4)).groupby('e').pre.sum(); print('\n조기 예측이 있는 실험 수', int((g > 0).sum()), '| 상위', g.sort_values(ascending=False).head(6).tolist())
