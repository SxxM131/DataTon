"""견고성 검증: 한 단계가 '같은 모양으로 K배 느리게' 진행되는 합성 실험 (구간 안을 선형 보간해 행을 K배로 늘림).

복제(늘리기)와 달리 신호가 톱니 모양이 되지 않는다. 라벨이 유효한 단계만 쓴다:
  - 5단계 → 6단계: 6단계 = 온도 합의 최고점이므로 느려져도 정답 위치가 그대로다.
  - 3단계 → 4단계: 4단계 = 전압 급락 사건이므로 3단계가 느려져도 그대로다.
(2단계는 '20초 안의 압력 상승'이라는 속도 기준이라 느리게 만들면 라벨이 달라진다 → 쓰지 않는다.)
공통 검증 rep 0, 모델은 원래 학습 fold로 학습. 6단계 근거 확인의 두 변형을 비교한다.
"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
NUM = [c for c in evalp.columns if c.startswith('meas_') or c in ('atmp_t', 'thermal_t', 'indu_t')]


def dilate(g, s, K):
    g = g.sort_values('relative_time_s').reset_index(drop=True); lab = g.label.to_numpy(int)
    if not (lab == s).any() or not (lab > s).any():
        return None
    a, b = int(np.argmax(lab == s)), int(np.argmax(lab > s))            # [a, b) = s단계, b = 다음 단계 첫 행
    if b - a < 3:
        return None
    if K == 1:
        out = g.copy(); out['src_id'] = out.sample_id.astype(str); return out
    dt = float(np.median(np.diff(g.relative_time_s)))
    if s == 5:
        pos = np.linspace(a, b, (b - a) * K + 1)[:-1]                    # 최고점(b)까지 이어지는 상승을 그대로 느리게
    else:
        pos = np.linspace(a, b - 1, (b - 1 - a) * K + 1)                 # 마지막 행(b−1)까지만 보간: 다음 단계로 넘어가는 사건(전압 급락)은 그대로 급격하게
    lo = np.floor(pos).astype(int); fr = pos - lo; hi = np.minimum(lo + 1, len(g) - 1)
    seg = g.iloc[lo].copy().reset_index(drop=True)
    for c in NUM:
        v = g[c].to_numpy(float); seg[c] = v[lo] * (1 - fr) + v[hi] * fr
    seg['label'] = s
    seg['relative_time_s'] = g.relative_time_s.iloc[a] + np.arange(len(seg)) * dt
    seg['src_id'] = g.sample_id.astype(str).to_numpy()[lo]
    seg['sample_id'] = [f'{sid}~{i}' for i, sid in enumerate(seg['src_id'])]
    head = g.iloc[:a].copy(); head['src_id'] = head.sample_id.astype(str)
    tail = g.iloc[b:].copy(); tail['src_id'] = tail.sample_id.astype(str); tail['relative_time_s'] = tail.relative_time_s + (len(seg) - (b - a)) * dt
    return pd.concat([head, seg, tail], ignore_index=True)


def gate6(p, frame, rate_thr):
    """6 예측인데 온도 합이 최고점을 갱신 중이면 5. rate_thr=None이면 갱신만 해도, 숫자면 초당 그만큼 넘게 오를 때만."""
    out = np.asarray(p).copy(); t_all = frame.relative_time_s.to_numpy(float); s_all = (frame.meas_t1 + frame.meas_t2).to_numpy(float); sid = frame.sample_id.astype(str).to_numpy()
    for _, idx in pd.Series(np.arange(len(frame))).groupby(frame.experiment_id.astype(str).to_numpy(), sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]; sm = s_all[idx]; cm = np.maximum.accumulate(sm)
        prev = np.r_[-np.inf, cm[:-1]]
        rate = np.r_[0.0, np.diff(sm) / np.diff(t_all[idx])]
        climbing = (sm > prev + 1e-9) if rate_thr is None else ((sm >= cm - 1e-9) & (rate > rate_thr))
        out[idx] = np.where((out[idx] == 6) & climbing, 5, out[idx])
    return out


def chain(a, pb, frame, rate_thr):
    x = M.gate_stage5(gate6(M.redecide_23(a, pb), frame, rate_thr), frame)
    return M.hold_stage(M.floor_stage4(x, frame), frame)


if __name__ == '__main__':
    yp = pool.label.to_numpy(int); PR = {**M.LGB_PARAMS, 'n_jobs': 6}; Xp = M.build_features(pool, color); res = []
    cfgname = lambda f: f.bat_class.astype(int).astype(str) + '/' + f.bat_cc.astype(str) + '/' + f.tr_method.astype(int).astype(str)  # noqa: E731
    for f in cv.make_folds(pool, n_repeats=1):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp[evalp.experiment_id.isin(f['valid_exps'])]
        er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
        a = lgb.LGBMClassifier(**PR).fit(Xp[tr], yp[tr]); b = lgb.LGBMClassifier(**PR).fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1), yp[tr])
        for s in (5, 3):
            for K in (1, 2, 3):
                fr = pd.concat([x for x in (dilate(g, s, K) for _, g in va.groupby('experiment_id', sort=True)) if x is not None], ignore_index=True)
                col = color.reindex(fr.src_id.values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
                X = M.build_features(fr, col); pa = a.predict(X); pb = b.predict_proba(pd.concat([X, M.pressure_features(fr, tab)], axis=1))
                for nm, thr in (('초당 0.2℃ 넘게 오를 때만 (현재)', 0.2), ('최고점을 갱신하기만 해도', None)):
                    res.append(pd.DataFrame({'s': s, 'K': K, 'm': nm, 'cfg': cfgname(fr).to_numpy(), 'y': fr.label.to_numpy(int), 'p': chain(pa, pb, fr, thr)}))
        print('fold', f['fold'], flush=True)
    d = pd.concat(res); rows = []
    for (s, K, m), x in d.groupby(['s', 'K', 'm']):
        f = f1_score(x.y, x.p, labels=range(1, 7), average=None, zero_division=0)
        rows.append({'느려진 단계': s, '배수': f'×{K}', '6단계 확인 방식': m, f'그 단계 행': int((x.y == s).sum()), '다음 단계 이상으로 조기 예측': int(((x.y == s) & (x.p > s)).sum()),
                     '6을 5 이하로': int(((x.y == 6) & (x.p <= 5)).sum()), '5단계 F1': round(f[4], 3), '6단계 F1': round(f[5], 3), 'Macro F1': round(f.mean(), 4)})
    pd.set_option('display.width', 230); print(pd.DataFrame(rows).to_string(index=False))
    d.to_pickle(C / 'robust_dilate.pkl')
