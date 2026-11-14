"""견고성 검증: 4단계(전압 급락 후 온도 급상승 전) 구간을 복제해 길게 늘린 합성 실험에서 5단계 조기 예측이 생기는지, 게이트가 막는지.

대상: 각형 28Ah 가열, 각형 280Ah 과충전 검증 실험 (공통 검증 rep 0, 모델은 원래 학습 fold로 학습).
늘리기: 4단계 행 중 앞 2행·뒤 1행을 뺀 블록을 K번 더 이어 붙이고 이후 시각을 그만큼 민다. 붙인 행의 라벨은 4 (급상승 전이므로 가이드 규칙상 4단계).
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
from surge_gate import surge_seen  # noqa: E402

C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
y = pool.label.to_numpy(int)
tgt = ((evalp.bat_class == 3) & (evalp.bat_cc == 28) & (evalp.tr_method == 1)) | ((evalp.bat_class == 3) & (evalp.bat_cc == 280) & (evalp.tr_method == 2))


def stretch(g, K):
    g = g.sort_values('relative_time_s').reset_index(drop=True)
    lab = g.label.to_numpy(int)
    if not ((lab == 4).any() and (lab == 5).any()):
        return None
    i4, i5 = int(np.argmax(lab == 4)), int(np.argmax(lab == 5))
    a, b = i4 + 2, i5 - 1
    if b - a < 2:
        return None
    dt = float(np.median(np.diff(g.relative_time_s)))
    blk = g.iloc[a:b]
    parts = [g.iloc[:b]]
    shift = 0.0
    for k in range(K):
        c = blk.copy(); shift += (blk.relative_time_s.iloc[-1] - blk.relative_time_s.iloc[0]) + dt
        c['relative_time_s'] = c.relative_time_s + shift; c['sample_id'] = c.sample_id.astype(str) + f'#{k}'
        parts.append(c)
    tail = g.iloc[b:].copy(); tail['relative_time_s'] = tail.relative_time_s + shift
    out = pd.concat(parts + [tail], ignore_index=True)
    out['src_id'] = out.sample_id.astype(str).str.split('#').str[0]
    return out


PR = {**M.LGB_PARAMS, 'n_jobs': 6}
Xp = M.build_features(pool, color)
res = []
for f in cv.make_folds(pool, n_repeats=1):
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
    va = evalp[evalp.experiment_id.isin(f['valid_exps']) & tgt]
    if va.empty:
        continue
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    a = lgb.LGBMClassifier(**PR).fit(Xp[tr], y[tr])
    b = lgb.LGBMClassifier(**PR).fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1), y[tr])
    for K in (0, 2, 4):
        frames = [stretch(g, K) for _, g in va.groupby('experiment_id', sort=True)]
        fr = pd.concat([x for x in frames if x is not None], ignore_index=True)
        col = color.reindex(fr.src_id.values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
        X = M.build_features(fr, col)
        p = M.redecide_23(a.predict(X), b.predict_proba(pd.concat([X, M.pressure_features(fr, tab)], axis=1)))
        for thr in (None, 12.0):
            q = p.copy()
            if thr is not None:
                q[(q == 5) & ~surge_seen(fr, thr, rate_thr=1.0)] = 4
            res.append(pd.DataFrame({'K': K, 'gate': '없음' if thr is None else f'{thr:.0f}℃', 'cfg': np.where(fr.bat_cc == 28, '각형 28Ah', '각형 280Ah 과충전'), 'e': fr.experiment_id, 'y': fr.label.to_numpy(int), 'p': q}))
    print('fold', f['fold'], flush=True)
d = pd.concat(res)
from sklearn.metrics import f1_score
rows = []
for (K, gate, cfg), g in d.groupby(['K', 'gate', 'cfg']):
    f = f1_score(g.y, g.p, labels=range(1, 7), average=None, zero_division=0)
    rows.append({'늘린 배수': f'×{K + 1}', '게이트': gate, '구성': cfg, '실험': g.e.nunique(), '4단계 행': int((g.y == 4).sum()), '4를 5로 예측': int(((g.y == 4) & (g.p == 5)).sum()),
                 '5를 4로 예측': int(((g.y == 5) & (g.p == 4)).sum()), '4단계 F1': round(f[3], 3), '5단계 F1': round(f[4], 3), 'Macro F1': round(f.mean(), 4)})
pd.set_option('display.width', 220)
print(pd.DataFrame(rows).sort_values(['구성', '늘린 배수', '게이트']).to_string(index=False))
