"""구성별로 보정한 '압력 상승 규칙'으로 2·3단계를 다시 정하는 방식의 공통 검증 (보정은 학습 fold 실험에서만).

규칙 계열: 온도 50~90℃ 조건에서 p(t) − p(t−w) 가 임계 이상이 된 적이 있으면 3단계.
  임계 종류 abs: 절대 압력 상승량,  rel: 초기 압력 대비 비율.  w ∈ {10, 20}초.  온도 조건 시점 lag(t−w) / cur(t).
구성(종류·용량·유발 방식 + 가이드상 특수 여부)마다 학습 실험의 2·3단계 오류 행 수가 최소인 (계열, 임계)를 고른다.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

FAMS = [(kind, w, gate) for kind in ('abs', 'rel') for w in (10.0, 20.0) for gate in ('lag', 'cur')]


def exp_arrays(g):
    """한 실험(Test식 행)의 계열별 단조 신호 M과 라벨 누적합."""
    g = g.sort_values('relative_time_s')
    t = g.relative_time_s.to_numpy(float); p = g.meas_p.to_numpy(float); t1 = g.meas_t1.to_numpy(float)
    p0 = max(abs(p[0]), 1e-3)
    M = {}
    for w in (10.0, 20.0):
        j = np.searchsorted(t, t - w, side='right') - 1; ok = j >= 0
        d = np.zeros(len(t)); d[ok] = p[ok] - p[j[ok]]
        t1b = np.full(len(t), np.nan); t1b[ok] = t1[j[ok]]
        for gate, cond in (('lag', ok & (t1b >= 50) & (t1b <= 90)), ('cur', ok & (t1 >= 50) & (t1 <= 90))):
            m = np.maximum.accumulate(np.where(cond, d, 0.0))
            M[('abs', w, gate)] = m
            M[('rel', w, gate)] = m / p0
    y = g.label.to_numpy(int) if 'label' in g else None
    return {'index': g.index, 'M': M, 'c2': np.concatenate([[0], np.cumsum(y == 2)]), 'c3': np.concatenate([[0], np.cumsum(y == 3)]), 'n': len(t)}


def err_rows(a, fam, thr):
    i = int(np.searchsorted(a['M'][fam], thr, side='left'))       # 처음 임계 이상이 되는 위치 (없으면 n)
    return (a['c2'][-1] - a['c2'][i]) + a['c3'][i]


def calibrate(arrs):
    """학습 실험들(arrs)의 총 오류 행이 최소인 (계열, 임계, 오류 행). 최소 구간의 기하 중앙을 임계로."""
    best = None
    for fam in FAMS:
        vals = np.concatenate([a['M'][fam][a['M'][fam] > 0] for a in arrs]) if arrs else np.array([])
        if len(vals) == 0:
            continue
        grid = np.exp(np.linspace(np.log(max(vals.min(), 1e-6)), np.log(vals.max()), 80))
        e = np.array([sum(err_rows(a, fam, th) for a in arrs) for th in grid])
        k = np.flatnonzero(e == e.min())
        run = k[: 1 + np.argmax(np.diff(k) > 1)] if (np.diff(k) > 1).any() else k      # 첫 연속 최소 구간
        th = float(np.sqrt(grid[run[0]] * grid[run[-1]]))
        if best is None or e.min() < best[2]:
            best = (fam, th, int(e.min()))
    return best


def cfg_key(frame):
    sp = ((frame.bat_class == 3) & (frame.bat_cc == 280) & (frame.tr_method == 1)) | ((frame.bat_class == 2) & (frame.bat_cc == 73) & (frame.tr_method == 1) & (frame.soc_status == 1))
    return frame.bat_class.astype(float).astype(str) + '/' + frame.bat_cc.astype(float).astype(str) + '/' + frame.tr_method.astype(float).astype(str) + np.where(sp, '*', '')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--save', default=''); ap.add_argument('--margin', type=float, default=1.0)
    args = ap.parse_args()
    pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
    R_ = HERE.parent / 'results' / 'oof'
    A = pd.read_pickle(R_ / 'rules_tab_nocum_r3.pkl').reindex(evalp.index)
    B = pd.read_pickle(R_ / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
    evalp = evalp.assign(cfg=cfg_key(evalp))
    arr = {e: exp_arrays(g) for e, g in evalp.groupby('experiment_id', sort=False)}
    cfg_of = evalp.groupby('experiment_id').cfg.first()
    y = evalp.label.to_numpy(int); pos = pd.Series(np.arange(len(evalp)), index=evalp.index)
    base = pd.DataFrame(index=evalp.index); ab = base.copy(); rule = base.copy(); gated = base.copy()
    log = []
    for r in range(3):
        a = A[f'pred_r{r}'].to_numpy(int)
        b = np.where(B[f'p3_r{r}'].to_numpy() > B[f'p2_r{r}'].to_numpy(), 3, 2)
        s23 = (a == 2) | (a == 3)
        base[f'pred_r{r}'] = a; ab[f'pred_r{r}'] = np.where(s23, b, a)
    mab = {r: ((ab[f'pred_r{r}'].to_numpy() != y) & ((y == 2) | (y == 3))) for r in range(3)}      # 모델(A+B)의 2·3단계 오류 행
    for c in (rule, gated):
        for r in range(3):
            c[f'pred_r{r}'] = ab[f'pred_r{r}'].to_numpy()
    for f in cv.make_folds(pool, n_repeats=3):
        r = f['rep']; col = f'pred_r{r}'
        tr_e, va_e = [e for e in f['train_exps'] if e in arr], [e for e in f['valid_exps'] if e in arr]
        for c in sorted(set(cfg_of[va_e])):
            tr_c = [e for e in tr_e if cfg_of[e] == c]
            if len(tr_c) < 3:
                continue
            fam, th, e_rule = calibrate([arr[e] for e in tr_c])
            e_model = int(sum(mab[r][pos[arr[e]['index']].to_numpy()].sum() for e in tr_c))        # 같은 학습 실험에서 모델의 OOF 오류 행
            use = e_rule < args.margin * e_model
            ev_r = ev_m = 0
            for e in (x for x in va_e if cfg_of[x] == c):
                ix = arr[e]['index']; ii = pos[ix].to_numpy()
                i = int(np.searchsorted(arr[e]['M'][fam], th, side='left'))
                pr = np.where(np.arange(arr[e]['n']) >= i, 3, 2)
                a = base[col].to_numpy()[ii]; s23 = (a == 2) | (a == 3)
                new = np.where(s23, pr, a)
                rule.iloc[ii, rule.columns.get_loc(col)] = new
                if use:
                    gated.iloc[ii, gated.columns.get_loc(col)] = new
                in23 = (y[ii] == 2) | (y[ii] == 3)
                ev_r += int(((new != y[ii]) & in23).sum()); ev_m += int(mab[r][ii].sum())
            log.append({'rep': r, 'fold': f['fold'], 'cfg': c, 'fam': f'{fam[0]}/{int(fam[1])}/{fam[2]}', 'thr': round(th, 4), 'n_tr': len(tr_c),
                        'tr_rule': e_rule, 'tr_model': e_model, 'use': use, 'va_rule': ev_r, 'va_model': ev_m})
    L = pd.DataFrame(log)
    pd.set_option('display.width', 250)
    s = L.groupby('cfg').agg(folds=('use', 'size'), used=('use', 'sum'), tr_rule=('tr_rule', 'sum'), tr_model=('tr_model', 'sum'), va_rule=('va_rule', 'sum'), va_model=('va_model', 'sum'))
    s['va_gated'] = L.assign(g=np.where(L.use, L.va_rule, L.va_model)).groupby('cfg').g.sum()
    s[['va_rule', 'va_model', 'va_gated']] = (s[['va_rule', 'va_model', 'va_gated']] / 3).round(0).astype(int)        # 3회 반복 평균
    print(s.to_string())
    print('\n구성별 선택된 규칙 (fold별):'); print(L.groupby('cfg').fam.agg(lambda x: x.value_counts().to_dict()).to_string())
    r0 = cv.evaluate(evalp, base, n_boot=200)
    print(f"\n기본 A              CV {r0['cv']:.4f} | per-class {np.round(r0['per_class'], 3).tolist()}")
    for nm, o in (('A+B (현재 v2)', ab), ('규칙 전면 적용', rule), ('규칙 선택 적용', gated)):
        rr = cv.evaluate(evalp, o, n_boot=300); c1 = cv.compare(evalp, o, base, n_boot=3000); c2 = cv.compare(evalp, o, ab, n_boot=3000)
        print(f"{nm:14s} CV {rr['cv']:.4f} [{rr['boot_ci90'][0]:.3f},{rr['boot_ci90'][1]:.3f}] | long {rr['long_exps']:.3f} short {rr['short_exps']:.3f} | per-class {np.round(rr['per_class'], 3).tolist()}"
              f" | vs A {c1['diff']:+.4f} p {c1['p_a_better']:.2f} | vs A+B {c2['diff']:+.4f} [{c2['ci90'][0]:+.3f},{c2['ci90'][1]:+.3f}] p {c2['p_a_better']:.2f}", flush=True)
        if args.save and nm == '규칙 선택 적용':
            cv.save_result(args.save, rr, o)
