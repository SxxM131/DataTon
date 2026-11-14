"""팀원(soomin) 변경안 검증: 테이블 모델의 이미지 피처를 10개(비율 5 + 직전 프레임 대비 변화 5)로 줄인 것. 공통 검증 3회.

팀원 코드(origin/soomin:model.py)의 build_features_img10 을 그대로 불러 쓴다. 기본 모델·재판정 모델 모두 다시 학습해 OOF 확률을 만든다.
"""
import importlib.util, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

spec = importlib.util.spec_from_file_location('soomin_model', sys.argv[1]); SM = importlib.util.module_from_spec(spec); spec.loader.exec_module(SM)
C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
full_cols = list(M.build_features(pool.head(50), color).columns)
non_img = [c for c in full_cols if not SM.is_image_col(c)]
print('전체 피처', len(full_cols), '| 이미지 피처 제외', len(non_img), '| 뺀 이미지 피처', len(full_cols) - len(non_img), flush=True)
Xp = SM.build_features_img10(pool, color, non_img); Xe = SM.build_features_img10(evalp, color, non_img)
print('img10 피처 수', Xp.shape[1], flush=True)
y = pool.label.to_numpy(int)
PA = np.zeros((3, len(evalp), 6), np.float32); PB = np.zeros_like(PA)
PR = {**M.LGB_PARAMS, 'n_jobs': 6}
for f in cv.make_folds(pool, n_repeats=3):
    r = f['rep']; tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    PA[r, va] = lgb.LGBMClassifier(**PR).fit(Xp[tr], y[tr]).predict_proba(Xe[va])
    Xb_tr = pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1)
    PB[r, va] = lgb.LGBMClassifier(**PR).fit(Xb_tr, y[tr]).predict_proba(pd.concat([Xe[va], M.pressure_features(evalp[va], tab)], axis=1))
    print('fold', r, f['fold'], flush=True)
np.savez_compressed(C / 'robust_img10_oof.npz', PA=PA, PB=PB, index=evalp.index.to_numpy())
