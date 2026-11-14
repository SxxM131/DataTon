"""model.py의 압력 상승 피처가 실험 코드(experiments/stage3/presrise.py)와 같은 값을 내는지 확인."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common'), str(HERE.parent / 'stage3')]
import cv, presrise  # noqa: E402
import model as M  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
er0, er1 = presrise.experiment_ranges(pool), M.pressure_ranges(pool)
t0, t1 = presrise.range_table(er0), M.range_table(er1)
l0, l1 = presrise.loo_ranges(er0), M.loo_ranges(er1)
assert l0 == l1 and abs(t0['all'] - t1['all']) < 1e-12 and len(t0['cfg']) == len(t1['cfg'])
for name, fr, own0, own1 in [('pool+LOO', pool, l0, l1), ('eval(table)', evalp, None, None), ('eval 섞음', evalp.sample(frac=1, random_state=0), None, None)]:
    a = presrise.features(fr, t0, own_range=own0, mask_special=True)
    b = M.pressure_features(fr, t1, own_range=own1)
    assert list(a.columns) == list(b.columns), '열 순서 다름'
    same = np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)
    print(f'{name:12s} 행 {len(fr)}, 열 {a.shape[1]}, 비운 행 {int(a.q_R.isna().sum())} → 동일: {same}')
    assert same
print('구성별 압력 범위 중앙값:', {k: round(v, 3) for k, v in t1['cfg'].items()})
