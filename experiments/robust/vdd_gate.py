"""원통형(4.8Ah·6Ah 가열) 5단계 근거 게이트의 검증: 공통 검증(3회) + 4단계 구간 늘리기.

가이드: 원통형의 5단계 = 4단계 이후 15초 안에 전압이 25% 이상 다시 떨어진 지점(하강 완료점).
근거(과거만): 첫 전압 급락 이후, 직전 15초(급락 이후 관측만) 최고 전압 대비 (1−θ)배 이하로 떨어진 적이 있다.
  첫 급락에서 전압이 이미 low_v 아래로 내려간 실험은 '두 번째 강하'를 볼 수 없으므로 근거가 있는 것으로 둔다 (게이트 미적용).
게이트: 원통형에서 5단계 예측인데 근거가 없으면 4단계.
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


def second_drop_seen(frame, theta=0.25, w=15.0, low_v=0.5):
    out = np.zeros(len(frame), bool)
    t_all = frame.relative_time_s.to_numpy(float); sid = frame.sample_id.astype(str).to_numpy(); v_all = frame.meas_v.to_numpy(float)
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t_all[idx]))]
        t, v = t_all[idx], v_all[idx]
        dv = np.r_[0.0, np.diff(v)]
        ev = (dv <= M.V_DROP) | (v < M.V_LOW)
        if not ev.any():
            continue
        ivd = int(np.argmax(ev)); n = len(t)
        lo = np.maximum(np.searchsorted(t, t - w, side='left'), ivd)          # 창의 시작 (급락 이후 관측만)
        hit = np.zeros(n, bool)
        for s in range(ivd + 1, n):
            if lo[s] < s:
                hit[s] = v[s] <= (1 - theta) * v[lo[s]:s].max()
        hit |= (np.arange(n) >= ivd) & (v < low_v)
        out[idx] = np.maximum.accumulate(hit)
    return out


def stretch(g, K):
    g = g.sort_values('relative_time_s').reset_index(drop=True); lab = g.label.to_numpy(int)
    if not ((lab == 4).any() and (lab == 5).any()):
        return None
    i4, i5 = int(np.argmax(lab == 4)), int(np.argmax(lab == 5)); a, b = i4 + 2, i5 - 1
    if b - a < 3:
        return None
    dt = float(np.median(np.diff(g.relative_time_s))); blk = g.iloc[a:b]; parts = [g.iloc[:b]]; shift = 0.0
    for k in range(K):
        c = blk.copy(); shift += (blk.relative_time_s.iloc[-1] - blk.relative_time_s.iloc[0]) + dt
        c['relative_time_s'] = c.relative_time_s + shift; c['sample_id'] = c.sample_id.astype(str) + f'#{k}'; parts.append(c)
    tail = g.iloc[b:].copy(); tail['relative_time_s'] = tail.relative_time_s + shift
    out = pd.concat(parts + [tail], ignore_index=True); out['src_id'] = out.sample_id.astype(str).str.split('#').str[0]
    return out


if __name__ == '__main__':
    C = Path.home() / 'jaehyeok' / 'cache'
    pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
    R = HERE.parent / 'results' / 'oof'
    A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
    y = evalp.label.to_numpy(int)
    base = pd.DataFrame(index=evalp.index)
    for r in range(3):
        base[f'pred_r{r}'] = M.gate_stage5(M.redecide_23(A[f'pred_r{r}'].to_numpy(int), B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy()), evalp)
    cyl = (evalp.bat_class == 1).to_numpy()
    r0 = cv.evaluate(evalp, base, n_boot=200)
    print(f"현재 v3(테이블) CV {r0['cv']:.4f} per-class {np.round(r0['per_class'], 3).tolist()}")
    # 근거가 처음 참이 되는 위치 (5단계 시작 행 대비)
    for th in (0.2, 0.25, 0.3):
        seen = second_drop_seen(evalp, th); pos = []
        for e, g in evalp[cyl].assign(seen=seen[cyl]).groupby('experiment_id'):
            g = g.sort_values('relative_time_s'); lab = g.label.to_numpy(int)
            if (lab == 5).any() and g.seen.any():
                pos.append(int(np.argmax(g.seen.to_numpy())) - int(np.argmax(lab == 5)))
        pos = np.array(pos)
        print(f"θ={th}: 근거 첫 발화 − 5단계 시작(행): ≤-3 {int((pos <= -3).sum())}, -2~-1 {int(((pos >= -2) & (pos <= -1)).sum())}, 0 {int((pos == 0).sum())}, +1 {int((pos == 1).sum())}, ≥+2 {int((pos >= 2).sum())} (n={len(pos)})")
        for mode in ('5→4', '5→4 & 4→5'):
            g = base.copy(); ch = []
            for r in range(3):
                p0 = base[f'pred_r{r}'].to_numpy(); p = p0.copy()
                m = cyl & (p == 5) & ~seen; p[m] = 4
                if mode != '5→4':
                    m2 = cyl & (p0 == 4) & seen; p[m2] = 5; m = m | m2
                ch.append((int(m.sum()), int((m & (p == y)).sum()), int((m & (p0 == y)).sum()))); g[f'pred_r{r}'] = p
            c = cv.compare(evalp, g, base, n_boot=3000); rr = cv.evaluate(evalp, g, n_boot=100); ch = np.mean(ch, 0)
            print(f"   [{mode:10s}] CV {rr['cv']:.4f} ({c['diff']:+.4f}, p {c['p_a_better']:.2f}) | 4·5단계 F1 {rr['per_class'][3]:.3f}/{rr['per_class'][4]:.3f} | 바뀐 행 {ch[0]:.0f} (바꿔서 맞음 {ch[1]:.0f}, 원래 맞았음 {ch[2]:.0f})", flush=True)
    # 4단계 구간 늘리기 (rep 0)
    color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
    yp = pool.label.to_numpy(int); PR = {**M.LGB_PARAMS, 'n_jobs': 6}; Xp = M.build_features(pool, color); res = []
    for f in cv.make_folds(pool, n_repeats=1):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp[evalp.experiment_id.isin(f['valid_exps']) & cyl]
        er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
        a = lgb.LGBMClassifier(**PR).fit(Xp[tr], yp[tr]); b = lgb.LGBMClassifier(**PR).fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1), yp[tr])
        for K in (0, 2, 4):
            fr = pd.concat([x for x in (stretch(g, K) for _, g in va.groupby('experiment_id', sort=True)) if x is not None], ignore_index=True)
            col = color.reindex(fr.src_id.values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
            X = M.build_features(fr, col); p = M.redecide_23(a.predict(X), b.predict_proba(pd.concat([X, M.pressure_features(fr, tab)], axis=1)))
            for gate in (None, 0.25):
                q = p.copy()
                if gate:
                    q[(q == 5) & ~second_drop_seen(fr, gate)] = 4
                res.append(pd.DataFrame({'K': K, 'gate': '없음' if gate is None else f'θ={gate}', 'y': fr.label.to_numpy(int), 'p': q, 'e': fr.experiment_id}))
    d = pd.concat(res)
    from sklearn.metrics import f1_score
    rows = []
    for (K, gate), g in d.groupby(['K', 'gate']):
        f = f1_score(g.y, g.p, labels=range(1, 7), average=None, zero_division=0)
        rows.append({'늘린 배수': f'×{K + 1}', '게이트': gate, '실험': g.e.nunique(), '4단계 행': int((g.y == 4).sum()), '4를 5로 예측': int(((g.y == 4) & (g.p == 5)).sum()), '5를 4로 예측': int(((g.y == 5) & (g.p == 4)).sum()),
                     '4단계 F1': round(f[3], 3), '5단계 F1': round(f[4], 3), 'Macro F1': round(f.mean(), 4)})
    print(pd.DataFrame(rows).to_string(index=False))
