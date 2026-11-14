"""기준 GBDT(ref_gbdt_color_extra)로 Test 예측 → 리더보드 보정용 제출 파일 (최종 제출용 아님: 최종은 test.py로 생성).

예) OMP_NUM_THREADS=6 python experiments/common/predict_ref_gbdt.py --data_dir ~/data/battery_data_light --cache_dir ~/jaehyeok/cache --output submissions/sub02_ref_gbdt.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]
import cv  # noqa: E402
from exp01_image_color import color_stats, COLOR_STATS  # noqa: E402
from exp02_gbdt import make_X, gbdt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', type=Path, required=True)
ap.add_argument('--cache_dir', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
groups = ['color', 'extra']
pool = cv.load_pool(args.data_dir)
color = pd.concat([pd.read_pickle(args.cache_dir / f'{s}_color.pkl') for s in ('train', 'validation')])
test = pd.read_csv(args.data_dir / 'test' / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
cache = args.cache_dir / 'test_color.pkl'
if not cache.exists():
    rows = Parallel(n_jobs=6, batch_size=64)(delayed(color_stats)(args.data_dir / 'test' / p) for p in test.image_path)
    pd.DataFrame(rows, columns=COLOR_STATS, index=pd.Index(test.sample_id.values, name='sample_id')).to_pickle(cache)
m = gbdt().fit(make_X(pool, color, groups), pool['label'].astype(int))      # Train+Validation 97개 실험 전체로 학습
pred = m.predict(make_X(test, pd.read_pickle(cache), groups)).astype(int)
args.output.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'sample_id': test.sample_id, 'label': pred}).to_csv(args.output, index=False)
print('저장', args.output, len(pred), pd.Series(pred).value_counts(normalize=True).sort_index().round(3).to_dict())
