"""최종 보정 체인 후보 비교: 공통 검증(3회) + 4단계 구간 늘리기(rep 0).

체인 A (현재 v5): 2·3 재판정 → 6단계 확인(오르는 중이면 5) → 5단계 확인(5인데 근거 없으면 4)
체인 B: A에서 5단계 확인을 '5 이상인데 근거 없으면 4'로 넓힘
체인 C: B → 4·5단계는 한 번 넘어가면 되돌아가지 않음
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
from hold_test import hold  # noqa: E402


def gate5_wide(p, frame):
    out = np.asarray(p).copy()
    for cfgs, seen_fn in ((M.SURGE_CFGS, M.surge_seen), (M.VDD_CFGS, M.second_drop_seen)):
        target = M._cfg_mask(frame, cfgs)
        if target.any():
            out[target & (out >= 5) & ~seen_fn(frame)] = 4
    return out


def chains(a, pb, frame):
    x = M.gate_stage6(M.redecide_23(a, pb), frame)
    A_ = M.gate_stage5(x, frame); B_ = gate5_wide(x, frame); C_ = hold(B_, frame)
    return {'A 현재(v5)': A_, 'B 근거 확인 확대': B_, 'C 확대 + 되돌아가지 않기': C_}


if __name__ == '__main__':
    C = Path.home() / 'jaehyeok' / 'cache'
    pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
    R = HERE.parent / 'results' / 'oof'
    A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
    y = evalp.label.to_numpy(int)
    out = {}
    for r in range(3):
        for k, v in chains(A[f'pred_r{r}'].to_numpy(int), B[[f'p{c}_r{r}' for c in range(1, 7)]].to_numpy(), evalp).items():
            out.setdefault(k, pd.DataFrame(index=evalp.index))[f'pred_r{r}'] = v
    cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
    intest = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])
    base = out['A 현재(v5)']
    for k, g in out.items():
        rr = cv.evaluate(evalp, g, n_boot=300); line = f"{k:24s} CV {rr['cv']:.4f} [{rr['boot_ci90'][0]:.3f},{rr['boot_ci90'][1]:.3f}] {np.round(rr['per_class'], 3).tolist()} | Test 크기 모의 {rr['test11_mean']:.3f}±{rr['test11_sd']:.3f}"
        if k != 'A 현재(v5)':
            c = cv.compare(evalp, g, base, n_boot=3000); c2 = cv.compare(evalp[intest], g[intest], base[intest], n_boot=2000)
            net = pd.Series(sum(((base[f'pred_r{r}'] != g[f'pred_r{r}']) & (g[f'pred_r{r}'] == y)).astype(int) - ((base[f'pred_r{r}'] != g[f'pred_r{r}']) & (base[f'pred_r{r}'] == y)).astype(int) for r in range(3)).to_numpy()).groupby(evalp.experiment_id.to_numpy()).sum()
            keep = (evalp.experiment_id != net.idxmax()).to_numpy(); c3 = cv.compare(evalp[keep], g[keep], base[keep], n_boot=1500)
            line += f" | vs A {c['diff']:+.4f} p {c['p_a_better']:.2f} | Test 구성만 {c2['diff']:+.4f} p {c2['p_a_better']:.2f} | 최대 이득 실험 제외 {c3['diff']:+.4f} p {c3['p_a_better']:.2f}"
        print(line, flush=True)
    # 구간 늘리기
    color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
    grp = np.where(evalp.bat_class == 1, '원통형', np.where((evalp.bat_class == 3) & (evalp.bat_cc == 28) & (evalp.tr_method == 1), '각형 28Ah', np.where((evalp.bat_class == 3) & (evalp.bat_cc == 280) & (evalp.tr_method == 2), '각형 280Ah 과충전', '')))
    ev2 = evalp.assign(grp=grp)
    yp = pool.label.to_numpy(int); PR = {**M.LGB_PARAMS, 'n_jobs': 6}; Xp = M.build_features(pool, color); res = []
    for f in cv.make_folds(pool, n_repeats=1):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = ev2[ev2.experiment_id.isin(f['valid_exps']) & (ev2.grp != '')]
        er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
        a = lgb.LGBMClassifier(**PR).fit(Xp[tr], yp[tr]); b = lgb.LGBMClassifier(**PR).fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1), yp[tr])
        for K in (0, 2, 4):
            fr = pd.concat([x for x in (stretch(g, K) for _, g in va.groupby('experiment_id', sort=True)) if x is not None], ignore_index=True)
            col = color.reindex(fr.src_id.values); col.index = pd.Index(fr.sample_id.astype(str).values, name='sample_id')
            X = M.build_features(fr, col)
            for nm, q in chains(a.predict(X), b.predict_proba(pd.concat([X, M.pressure_features(fr, tab)], axis=1)), fr).items():
                res.append(pd.DataFrame({'K': K, 'm': nm, 'grp': fr.grp.to_numpy(), 'y': fr.label.to_numpy(int), 'p': q}))
    d = pd.concat(res); rows = []
    for (g_, K, m), x in d.groupby(['grp', 'K', 'm']):
        f = f1_score(x.y, x.p, labels=range(1, 7), average=None, zero_division=0)
        rows.append({'구성': g_, '늘린 배수': f'×{K + 1}', '방식': m, '4를 5 이상으로': int(((x.y == 4) & (x.p >= 5)).sum()), '4를 3 이하로': int(((x.y == 4) & (x.p <= 3)).sum()), '5를 4 이하로': int(((x.y == 5) & (x.p <= 4)).sum()), 'Macro F1': round(f.mean(), 4)})
    pd.set_option('display.width', 220); print(pd.DataFrame(rows).to_string(index=False))
