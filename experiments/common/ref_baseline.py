"""베이스라인 v7(센서 5개 + 그레이스케일 통계 + 로지스틱 회귀)을 공통 검증 체계로 채점 → 리더보드 Public 점수와 비교용.

예) OMP_NUM_THREADS=4 python experiments/common/ref_baseline.py --data_dir ~/data/battery_data_light --cache_dir ~/jaehyeok/cache
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import cv  # noqa: E402
import model as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', type=Path, required=True)
ap.add_argument('--cache_dir', type=Path, required=True)
args = ap.parse_args()
pool = cv.load_pool(args.data_dir)
evalp = cv.to_test_like(pool)
gray = pd.concat([pd.read_pickle(args.cache_dir / f'{s}_gray.pkl') for s in ('train', 'validation')])
X_fit, X_eval, y = M.build_features(pool, gray), M.build_features(evalp, gray), pool['label'].astype(int)


def run(f):
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
    va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    return evalp.index[va], M.build_model(None).fit(X_fit[tr], y[tr]).predict(X_eval[va])


oof = pd.DataFrame(index=evalp.index)
for f in cv.make_folds(pool, n_repeats=1):
    idx, p = run(f)
    oof.loc[idx, 'pred_r0'] = p
res = cv.evaluate(evalp, oof.astype(int))
idx, p = run(cv.official_split(pool))
res['official_val'] = cv.macro_f1(evalp.loc[idx, 'label'], p)
res['official_val_raw_rows'] = 0.6778
res['public_lb'] = 0.8516755528
cv.save_result('ref_baseline_v7', res, oof.astype(int))
print(cv.summary_line('ref_baseline_v7', res))
print(f"  Test 크기 모의: {res['test11_mean']:.3f} ± {res['test11_sd']:.3f} ({res['test11_q05_q95'][0]:.3f}~{res['test11_q05_q95'][1]:.3f})")
