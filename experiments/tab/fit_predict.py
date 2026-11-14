"""세션 A 추천 모델을 Train+Validation 97개 실험 전체로 학습하고 입력 CSV를 예측 (Test는 추론만, fit 없음).

추천: --feat all --cfg '{"cum":false}' (LGB: lr 0.1, 200트리, colsample 0.3, max_bin 63, n_jobs 6)
대체(센서 5개+이미지만): --feat sensor5 --cfg '{"cum":false,"rule":true}'

예)
  OMP_NUM_THREADS=6 python experiments/tab/fit_predict.py --input ~/data/battery_data_light/test/test.csv \
      --image_root ~/data/battery_data_light/test --output ~/jaehyeok/cache/tab/pred_tab_all_nocum.csv
출력: sample_id,label (입력 행 순서) + 같은 이름 _proba.csv (p1..p6)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / 'common'), str(HERE.parent.parent)]
import cv  # noqa: E402
import feats  # noqa: E402
from run import LGB_DEFAULT, DATA, CACHE  # noqa: E402
from exp01_image_color import color_stats, COLOR_STATS  # noqa: E402

FAST = dict(learning_rate=0.1, n_estimators=200, colsample_bytree=0.3, max_bin=63)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--image_root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--feat', default='all', choices=['all', 'sensor5'])
    ap.add_argument('--cfg', default='{"cum":false}')
    ap.add_argument('--seeds', type=int, default=1)
    args = ap.parse_args()
    cfg = {'extra': args.feat == 'all', **json.loads(args.cfg)}

    # 학습: 97개 실험 전체 원본 행 (이미지 색 통계는 캐시 사용)
    pool = cv.load_pool(DATA)
    color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
    X = feats.build(pool, color, cfg)
    y = pool['label'].astype(int).to_numpy()
    models = [lgb.LGBMClassifier(**{**LGB_DEFAULT, **FAST}, random_state=42 + s).fit(X, y) for s in range(args.seeds)]

    # 추론: 입력 CSV 행만으로 피처 생성 (실험별·인과적)
    te = pd.read_csv(args.input, dtype={'sample_id': 'string', 'experiment_id': 'string'})
    paths = [args.image_root / p for p in te['image_path'].astype(str)]
    rows = Parallel(n_jobs=6, batch_size=64)(delayed(color_stats)(p) for p in paths)
    tcol = pd.DataFrame(rows, columns=COLOR_STATS, index=pd.Index(te['sample_id'].astype(str).values, name='sample_id'))
    Xt = feats.build(te, tcol, cfg)[list(X.columns)]
    P = np.mean([m.predict_proba(Xt) for m in models], axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'sample_id': te['sample_id'], 'label': np.argmax(P, 1) + 1}).to_csv(args.output, index=False)
    pd.DataFrame(P, columns=[f'p{k}' for k in range(1, 7)]).assign(sample_id=te['sample_id'].values) \
        .to_csv(args.output.with_name(args.output.stem + '_proba.csv'), index=False)
    print(f'{len(te)}행 예측 → {args.output}', pd.Series(np.argmax(P, 1) + 1).value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
