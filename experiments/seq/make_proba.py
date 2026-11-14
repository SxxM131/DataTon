"""공통 fold(cv.make_folds)에서 LightGBM OOF 클래스 확률을 만들어 캐시한다 (디코더 실험 입력용).

- 피처: exp02_gbdt.make_X(frame, color, ['color','extra']) (기준 모델과 동일)
- 학습: pool 행 중 train_exps / 예측: evalp 행 중 valid_exps (피처는 evalp 행만 보고 생성)
- --inner: 각 outer fold의 train_exps 안에서 다시 4-fold → 학습 fold 내부 OOF 확률(디코더 하이퍼파라미터 튜닝용)
  (inner 예측도 evalp 행 = Test와 같은 다운샘플 행에 대해 만든다)

출력 (~/jaehyeok/cache/seq/):
  proba_outer.pkl : {'index': evalp.index, 'proba': {rep: (n_evalp, 6) 배열}}
  proba_inner.pkl : {(rep, fold): {'index': evalp 인덱스 배열, 'proba': (n, 6)}}
예) OMP_NUM_THREADS=4 python experiments/seq/make_proba.py --inner
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common'), str(HERE.parent), str(HERE.parent.parent)]
import cv  # noqa: E402
from exp02_gbdt import make_X  # noqa: E402

CACHE = Path.home() / 'jaehyeok' / 'cache'
OUT = CACHE / 'seq'


def lgbm():
    # 기준 HGB(lr 0.1, 200회, 잎 31, min_leaf 50, l2 1, max_features 0.8)와 최대한 같은 설정
    return lgb.LGBMClassifier(learning_rate=0.1, n_estimators=200, num_leaves=31, min_child_samples=50,
                              reg_lambda=1.0, colsample_bytree=0.8, subsample=1.0, n_jobs=4,
                              random_state=42, verbose=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path, default=Path.home() / 'data' / 'battery_data_light')
    ap.add_argument('--n_repeats', type=int, default=3)
    ap.add_argument('--inner', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pool = cv.load_pool(args.data_dir)
    evalp = cv.to_test_like(pool)
    xf = OUT / 'X_cache.pkl'
    if xf.exists():
        X_fit, X_eval = pd.read_pickle(xf)
    else:
        color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])
        X_fit, X_eval = make_X(pool, color, ['color', 'extra']), make_X(evalp, color, ['color', 'extra'])
        pd.to_pickle((X_fit, X_eval), xf)
    y = pool['label'].astype(int).to_numpy()
    print(f'pool {len(pool)} / evalp {len(evalp)} / 피처 {X_fit.shape[1]}', flush=True)

    def fit_pred(train_exps, valid_exps):
        tr = pool.experiment_id.isin(train_exps).to_numpy()
        va = evalp.experiment_id.isin(valid_exps).to_numpy()
        m = lgbm().fit(X_fit[tr], y[tr])
        assert list(m.classes_) == cv.LABELS
        return evalp.index[va].to_numpy(), m.predict_proba(X_eval[va])

    outer = {r: np.full((len(evalp), 6), np.nan) for r in range(args.n_repeats)}
    pos = pd.Series(np.arange(len(evalp)), index=evalp.index)
    inner = {}
    t0 = time.time()
    for f in cv.make_folds(pool, n_repeats=args.n_repeats):
        idx, p = fit_pred(f['train_exps'], f['valid_exps'])
        outer[f['rep']][pos[idx].to_numpy()] = p
        if args.inner:
            sub = pool[pool.experiment_id.isin(f['train_exps'])]
            ii, pp = [], []
            for g in cv.make_folds(sub, n_splits=4, n_repeats=1, seed=100 + f['rep']):
                a, b = fit_pred(g['train_exps'], g['valid_exps'])
                ii.append(a); pp.append(b)
            inner[(f['rep'], f['fold'])] = {'index': np.concatenate(ii), 'proba': np.concatenate(pp)}
        print(f"rep {f['rep']} fold {f['fold']} 완료 ({time.time() - t0:.0f}s)", flush=True)
        with open(OUT / 'proba_outer.pkl', 'wb') as fh:
            pickle.dump({'index': evalp.index.to_numpy(), 'proba': outer}, fh)
        if args.inner:
            with open(OUT / 'proba_inner.pkl', 'wb') as fh:
                pickle.dump(inner, fh)


if __name__ == '__main__':
    main()
