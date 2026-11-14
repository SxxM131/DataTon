"""공통 검증 체계의 기준 모델(reference) + 검증 체계 자체의 흔들림 측정.

기준 모델: exp02의 'base+color+extra' 피처 + HistGradientBoosting.
모든 트랙은 이 스크립트의 구조(피처는 pool / evalp 각각에서 따로 생성 → fold 루프 → evaluate)를 따라 하면 된다.

예) OMP_NUM_THREADS=6 python experiments/common/ref_gbdt.py --data_dir ~/data/battery_data_light --cache_dir ~/jaehyeok/cache
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]
import cv  # noqa: E402
import model as M  # noqa: E402
from exp02_gbdt import make_X, gbdt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--cache_dir', type=Path, required=True)
    ap.add_argument('--groups', default='color,extra')
    ap.add_argument('--name', default='ref_gbdt_color_extra')
    ap.add_argument('--n_repeats', type=int, default=3)
    args = ap.parse_args()
    groups = [g for g in args.groups.split(',') if g]

    pool = cv.load_pool(args.data_dir)
    evalp = cv.to_test_like(pool)
    color = pd.concat([pd.read_pickle(args.cache_dir / f'{s}_color.pkl') for s in ('train', 'validation')])
    X_fit = make_X(pool, color, groups)        # 학습 행: 전체 5초 간격
    X_eval = make_X(evalp, color, groups)      # 채점 행: Test처럼 다운샘플된 행만 보고 피처 생성
    y = pool['label'].astype(int)
    print(f'pool {len(pool)}행 / evalp {len(evalp)}행 / 실험 {pool.experiment_id.nunique()}개 / 피처 {X_fit.shape[1]}개', flush=True)

    def run(f):
        tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
        va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        m = gbdt().fit(X_fit[tr], y[tr])
        return evalp.index[va], m.predict(X_eval[va])

    oof = pd.DataFrame(index=evalp.index)
    for f in cv.make_folds(pool, n_repeats=args.n_repeats):
        idx, p = run(f)
        oof.loc[idx, f"pred_r{f['rep']}"] = p
    res = cv.evaluate(evalp, oof.astype(int))
    idx, p = run(cv.official_split(pool))
    res['official_val'] = cv.macro_f1(evalp.loc[idx, 'label'], p)
    cv.save_result(args.name, res, oof.astype(int), extra={'groups': groups})
    print(cv.summary_line(args.name, res))
    print(f"  Test 크기(11개 실험) 모의: 평균 {res['test11_mean']:.3f} ± {res['test11_sd']:.3f} (5~95%: {res['test11_q05_q95'][0]:.3f}~{res['test11_q05_q95'][1]:.3f})"
          f" | Public(30%)-Private(70%) 차이의 표준편차 {res['public_minus_private_sd']:.3f}")


if __name__ == '__main__':
    main()
