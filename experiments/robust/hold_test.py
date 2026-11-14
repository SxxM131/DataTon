"""'4·5단계는 한 번 넘어가면 되돌아가지 않는다'(hold)의 견고성: 4단계 구간을 늘린 합성 실험에서 조기 5단계가 굳어지지 않는지.

대상: 원통형, 각형 28Ah 가열, 각형 280Ah 과충전 (5단계 근거 확인이 있는 구성). 공통 검증 rep 0.
"""
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
from vdd_gate import stretch  # noqa: E402


def hold(p, frame, levels=(4, 5)):
    q = np.asarray(p).copy()
    t = frame.relative_time_s.to_numpy(float); sid = frame.sample_id.astype(str).to_numpy()
    for _, idx in pd.Series(np.arange(len(frame))).groupby(frame.experiment_id.astype(str).to_numpy(), sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t[idx]))]
        x = q[idx]
        for s in levels:
            x = np.where(np.maximum.accumulate(x >= s) & (x < s), s, x)
        q[idx] = x
    return q


if __name__ == '__main__':
    C = Path.home() / 'jaehyeok' / 'cache'
    pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
    color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
    grp = np.where(evalp.bat_class == 1, '원통형', np.where((evalp.bat_class == 3) & (evalp.bat_cc == 28) & (evalp.tr_method == 1), '각형 28Ah',
                   np.where((evalp.bat_class == 3) & (evalp.bat_cc == 280) & (evalp.tr_method == 2), '각형 280Ah 과충전', '')))
    evalp = evalp.assign(grp=grp)
    yp = pool.label.to_numpy(int); PR = {**M.LGB_PARAMS, 'n_jobs': 6}; Xp = M.build_features(pool, color); res = []
    for f in cv.make_folds(pool, n_repeats=1):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp[evalp.experiment_id.isin(f['valid_exps']) & (evalp.grp != '')]
        er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
        a = lgb.LGBMClassifier(**PR).fit(Xp[tr], yp[tr]); b = lgb.LGBMClassifier(**PR).fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1), yp[tr])
        for K in (0, 2, 4):
            fr = pd.concat([x for x in (stretch(g, K) for _, g in va.groupby('experiment_id', sort=True)) if x is not None], ignore_index=True)
            col = color.reindex(fr.src_id.values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
            X = M.build_features(fr, col)
            p = M.apply_rules(a.predict(X), b.predict_proba(pd.concat([X, M.pressure_features(fr, tab)], axis=1)), fr)
            for nm, q in (('근거 확인만', p), ('+ 되돌아가지 않기', hold(p, fr))):
                res.append(pd.DataFrame({'K': K, 'm': nm, 'grp': fr.grp.to_numpy(), 'y': fr.label.to_numpy(int), 'p': q}))
        print('fold', f['fold'], flush=True)
    d = pd.concat(res); rows = []
    for (g_, K, m), x in d.groupby(['grp', 'K', 'm']):
        f = f1_score(x.y, x.p, labels=range(1, 7), average=None, zero_division=0)
        rows.append({'구성': g_, '늘린 배수': f'×{K + 1}', '방식': m, '4를 5 이상으로': int(((x.y == 4) & (x.p >= 5)).sum()), '4를 3 이하로': int(((x.y == 4) & (x.p <= 3)).sum()), '5를 4 이하로': int(((x.y == 5) & (x.p <= 4)).sum()),
                     '4단계 F1': round(f[3], 3), '5단계 F1': round(f[4], 3), 'Macro F1': round(f.mean(), 4)})
    pd.set_option('display.width', 220); print(pd.DataFrame(rows).to_string(index=False))
