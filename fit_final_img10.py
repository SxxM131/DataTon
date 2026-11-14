"""선택안 C 최종 가중치: Train+Val 97실험 fit → weights/tab_lgbm.joblib, test 제출 CSV.

CNN은 기존 weights/cnn_pairdiff.pt 재사용 (신규 학습 없음).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import joblib
import numpy as np
import pandas as pd

import model as M

ROOT = Path(__file__).resolve().parent
DATA = Path.home() / 'data' / 'battery_data_light'
ART = ROOT / 'artifacts'
FUSE_W = 0.1


def load_pool() -> pd.DataFrame:
    frames = []
    for split in ('train', 'validation'):
        f = pd.read_csv(DATA / split / f'{split}.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
        f['origin'] = split
        f['image_file'] = M.resolve_image_paths(f, [DATA / split, DATA]).astype(str).values
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    t0 = time.time()
    meta = joblib.load(ART / 'tab_train_only_img10.joblib')
    non_img = list(meta['non_img'])
    cols = list(meta['columns'])
    cols23 = list(meta['columns23'])

    pool = load_pool()
    color = joblib.load(ART / 'color_features.joblib')
    if 'sample_id' in getattr(color, 'columns', []):
        color = color.set_index('sample_id')
    color.index = color.index.astype(str)

    print('build img10 features…', flush=True)
    X = M.build_features_img10(pool, color, non_img)[cols]
    y = pool['label'].to_numpy(int)
    print(f'fit LightGBM on {len(pool)} rows / {pool.experiment_id.nunique()} exps…', flush=True)
    m = M.build_model().fit(X, y)
    er = M.pressure_ranges(pool)
    table = M.range_table(er)
    Q = M.pressure_features(pool, table, own_range=M.loo_ranges(er))
    m23 = M.build_model().fit(pd.concat([X, Q], axis=1)[cols23], y)

    bundle = {
        'model': m,
        'columns': cols,
        'model23': m23,
        'columns23': cols23,
        'range_table': table,
        'fuse_w': FUSE_W,
        'feature_mode': 'img10',
        'non_img': non_img,
        'img10_names': list(meta['img10_names']),
        'train_rows': int(len(pool)),
        'train_experiments': int(pool.experiment_id.nunique()),
        'selection': 'C_img10_fuse_w0.1',
        'lgb_params': dict(M.LGB_PARAMS),
    }
    out_w = ROOT / 'weights' / 'tab_lgbm.joblib'
    # keep cnn_pairdiff.pt as-is (full, no retrain)
    joblib.dump(bundle, out_w)
    print(f'wrote {out_w} ({time.time()-t0:.1f}s)', flush=True)

    # test submission via test.py logic inline to verify
    import subprocess, sys
    sub = ROOT / 'submissions' / 'img10_fuse01_v7gates.csv'
    cmd = [
        sys.executable, str(ROOT / 'test.py'),
        '--input', str(DATA / 'test' / 'test.csv'),
        '--image_root', str(DATA / 'test'),
        '--output', str(sub),
        '--sample_submission', str(DATA / 'sample_submission.csv'),
        '--device', 'cuda',
    ]
    print('running', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    raise SystemExit(r.returncode)


if __name__ == '__main__':
    main()
