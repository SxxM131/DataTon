"""[라운드 2-2] tab_all_nocum_r3 설정(세션 A) 위에 이벤트 피처 추가 — 3반복, 같은 fold. 시작 행(onset) 정확도 포함.

python tab2_runs.py            # base 재실행 + ev 피처
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import imglib as L
import evfeat
from imglib import cv

sys.path.insert(0, str(L.HERE.parent / 'tab'))
import feats  # noqa: E402

pool, evalp = L.load_all()
color = pd.concat([pd.read_pickle(L.CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
CFG = {'extra': True, 'cum': False}
Xp, Xe = feats.build(pool, color, CFG), feats.build(evalp, color, CFG)
Ep, Ee = evfeat.get(pool, evalp)
y = pool.label.astype(int).to_numpy()
PARAMS = dict(objective='multiclass', learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.3, reg_lambda=1.0, n_jobs=L.NT, verbose=-1, deterministic=True, force_row_wise=True, max_bin=63, random_state=42)
NREP = 3


def onset_rows(evalp):
    """evalp(Test-like) 문맥에서 각 실험의 단계 s 첫 행 위치 (evalp 행 위치)."""
    out = {4: [], 5: [], 6: []}
    t = evalp.relative_time_s.to_numpy(); lab = evalp.label.to_numpy()
    for _, idx in evalp.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind='mergesort')]
        for s in out:
            if (lab[idx] == s).any():
                out[s].append(idx[np.argmax(lab[idx] == s)])
    return {s: np.array(v) for s, v in out.items()}


ON = onset_rows(evalp)


def onset_acc(oof):
    cols = [c for c in oof.columns if c.startswith('pred_r')]
    yy = evalp.label.to_numpy()
    return {s: float(np.mean([(oof[c].to_numpy()[r] == yy[r]).mean() for c in cols])) for s, r in ON.items()}


def run(name, A, B, keep_proba=False):
    oof = pd.DataFrame(index=evalp.index)
    for f in cv.make_folds(pool, n_repeats=NREP):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        m = lgb.LGBMClassifier(**PARAMS).fit(A[tr], y[tr])
        Pm = m.predict_proba(B[va]); r = f['rep']
        oof.loc[evalp.index[va], f'pred_r{r}'] = Pm.argmax(1) + 1
        for k in range(6):
            oof.loc[evalp.index[va], f'p{k + 1}_r{r}'] = Pm[:, k]
    for c in [c for c in oof.columns if c.startswith('pred_r')]:
        oof[c] = oof[c].astype(int)
    res = cv.evaluate(evalp, oof)
    cv.save_result(name, res, oof)
    print(cv.summary_line(name, res), flush=True)
    oa = onset_acc(oof)
    print(f"    시작 행 정확도 (evalp 문맥 첫 행, 4/5/6: n={[len(ON[s]) for s in (4, 5, 6)]}): {[round(oa[s], 3) for s in (4, 5, 6)]}", flush=True)
    return oof


if __name__ == '__main__':
    ref = pd.read_pickle(L.HERE.parent / 'results/oof/tab_all_nocum_r3.pkl')
    oa = onset_acc(ref)
    print(cv.summary_line('tab_all_nocum_r3 (세션 A OOF)', cv.evaluate(evalp, ref)), f"\n    시작 행 정확도 4/5/6: {[round(oa[s], 3) for s in (4, 5, 6)]}", flush=True)
    sets = {'img2_base_nocum': [], 'img2_ev12': list(Ep.columns), 'img2_ev4': ['ev_fprev', 'ev_fprev_lag1', 'ev_since_first', 'ev_since_last']}
    todo = sys.argv[1:] or list(sets)
    oofs = {}
    for name in todo:
        c = sets[name]
        A = pd.concat([Xp, Ep[c]], axis=1) if c else Xp
        B = pd.concat([Xe, Ee[c]], axis=1) if c else Xe
        oofs[name] = run(name, A, B)
        L.cmp(evalp, oofs[name], ref, name, 'tab_all_nocum_r3')
        if name != 'img2_base_nocum' and 'img2_base_nocum' in oofs:
            L.cmp(evalp, oofs[name], oofs['img2_base_nocum'], name, 'img2_base_nocum(재실행)')
