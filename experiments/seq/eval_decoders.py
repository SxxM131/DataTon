"""인과 단조 디코더 비교 (공통 검증 체계).

입력: make_proba.py가 만든 OOF 확률 (~/jaehyeok/cache/seq/proba_outer.pkl, proba_inner.pkl) 또는 --oof <pkl>(p{c}_r{rep} 열).
- 디코더 파라미터(CTMC 전이율·사전확률)는 각 fold의 학습 실험 라벨로만 추정.
- 'fixed' 행: 하이퍼파라미터를 미리 정해 둔 값으로 고정.
- 'nested' 행: 각 fold에서 학습 실험 내부 OOF 확률(proba_inner)로 가장 좋은 설정을 고른 뒤 평가 fold에 적용 (평가 fold로 튜닝하지 않음).

예) python experiments/seq/eval_decoders.py
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'common')]
import cv  # noqa: E402
from decode import decode, fit_params  # noqa: E402

CACHE = Path.home() / 'jaehyeok' / 'cache' / 'seq'
CONFIGS = {
    'argmax': dict(method='argmax'),
    'cummax': dict(method='cummax'),
    'persist15': dict(method='persist', persist_s=15),
    'persist30': dict(method='persist', persist_s=30),
    'ordinal.5': dict(method='ordinal', theta=0.5),
    'hmm_a1': dict(method='hmm', alpha=1.0),
    'hmm_a.5': dict(method='hmm', alpha=0.5),
    'hmm_a.3': dict(method='hmm', alpha=0.3),
    'hmm_a.2': dict(method='hmm', alpha=0.2),
    'hmm_a.5_noclamp': dict(method='hmm', alpha=0.5, clamp=False),
    'hmm_a.3_noclamp': dict(method='hmm', alpha=0.3, clamp=False),
    'hmm_a.5_b.3': dict(method='hmm', alpha=0.5, beta=0.3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, default=Path.home() / 'data' / 'battery_data_light')
    ap.add_argument('--oof', type=Path, default=None, help='p{c}_r{rep} 열이 있는 OOF pkl (없으면 proba_outer.pkl)')
    ap.add_argument('--tag', default='seq')
    ap.add_argument('--no_nested', action='store_true')
    args = ap.parse_args()

    pool = cv.load_pool(args.data_dir)
    evalp = cv.to_test_like(pool)
    pos = pd.Series(np.arange(len(evalp)), index=evalp.index)
    t_all, e_all, y_all = evalp.relative_time_s.to_numpy(float), evalp.experiment_id.to_numpy().astype(str), evalp.label.to_numpy(int)
    if args.oof is None:
        d = pickle.load(open(CACHE / 'proba_outer.pkl', 'rb'))
        assert (d['index'] == evalp.index.to_numpy()).all()
        outer = d['proba']
    else:
        o = pd.read_pickle(args.oof).reindex(evalp.index)
        reps = sorted({int(c.split('_r')[1]) for c in o.columns if c.startswith('p1_r')})
        outer = {r: o[[f'p{c}_r{r}' for c in cv.LABELS]].to_numpy() for r in reps}
    inner = None if (args.no_nested or args.oof is not None) else pickle.load(open(CACHE / 'proba_inner.pkl', 'rb'))
    folds = [f for f in cv.make_folds(pool) if f['rep'] in outer]

    oofs = {k: pd.DataFrame(index=evalp.index) for k in list(CONFIGS) + (['nested'] if inner else [])}
    chosen = []
    for f in folds:
        trp = pool[pool.experiment_id.isin(f['train_exps'])]
        params = fit_params(trp.label.to_numpy(), trp.relative_time_s.to_numpy(), trp.experiment_id.to_numpy())
        va = np.isin(e_all, f['valid_exps'].astype(str))
        P = outer[f['rep']][va]
        col = f"pred_r{f['rep']}"
        for k, kw in CONFIGS.items():
            oofs[k].loc[evalp.index[va], col] = decode(P, t_all[va], e_all[va], params, **kw)
        if inner:
            ii = inner[(f['rep'], f['fold'])]
            ip = pos[ii['index']].to_numpy()
            # 내부 OOF는 학습 실험 안에서 만든 것 → 여기서 고른 설정은 평가 fold를 보지 않는다
            sc = {k: cv.macro_f1(y_all[ip], decode(ii['proba'], t_all[ip], e_all[ip], params, **kw)) for k, kw in CONFIGS.items()}
            best = max(sc, key=sc.get)
            chosen.append(best)
            oofs['nested'].loc[evalp.index[va], col] = oofs[best].loc[evalp.index[va], col]
    res = {}
    for k, o in oofs.items():
        o = o.astype(int)
        res[k] = cv.evaluate(evalp, o, n_boot=500)
        c = cv.compare(evalp, o, oofs['argmax'].astype(int), n_boot=2000) if k != 'argmax' else None
        line = (f"{k:18s} CV {res[k]['cv']:.4f} [{res[k]['boot_ci90'][0]:.3f},{res[k]['boot_ci90'][1]:.3f}] | long {res[k]['long_exps']:.3f} short {res[k]['short_exps']:.3f}"
                f" | per-class {np.round(res[k]['per_class'], 3).tolist()}")
        if c:
            line += f" | vs argmax {c['diff']:+.4f} [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p {c['p_a_better']:.2f}"
        print(line, flush=True)
        cv.save_result(f'{args.tag}_{k}', res[k], o)
    if chosen:
        print('nested 선택:', pd.Series(chosen).value_counts().to_dict())


if __name__ == '__main__':
    main()
