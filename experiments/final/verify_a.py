"""검증 A: 루트 model.py 피처 == 실험 코드(tab/feats.build + rules.rule_features), 공통 CV(rep 0) 재현.

색 통계는 캐시(~/jaehyeok/cache/{train,validation}_color.pkl)를 쓴다 (검증 속도용; train.py/test.py는 이미지에서 직접 계산).
추가로 무작위 이미지 300장에 대해 model.color_stats == 캐시 값인지 확인.
예) OMP_NUM_THREADS=6 python experiments/final/verify_a.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments'
sys.path[:0] = [str(ROOT), str(EXP / 'common'), str(EXP / 'tab'), str(EXP / 'rules')]
import model as M  # noqa: E402
import cv  # noqa: E402
import feats  # noqa: E402
import rules  # noqa: E402

DATA = Path.home() / 'data/battery_data_light'
CACHE = Path.home() / 'jaehyeok/cache'


def same(a: pd.DataFrame, b: pd.DataFrame, name):
    assert list(a.columns) == list(b.columns), f'{name}: 열 불일치'
    assert a.index.equals(b.index), f'{name}: 인덱스 불일치'
    eq = np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True)
    print(f'  {name}: shape {a.shape}, 값 완전 일치 {eq}', flush=True)
    return eq


t0 = time.time()
pool = cv.load_pool(DATA)
evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(CACHE / f'{s}_color.pkl') for s in ('train', 'validation')])

# (0) 색 통계 일치 (무작위 300장)
rng = np.random.default_rng(0)
pick = pool.iloc[rng.choice(len(pool), 300, replace=False)]
c_new = M.extract_image_features(pick, pick['image_file'].tolist())
print('  color_stats 300장 vs 캐시 완전 일치:', np.array_equal(c_new.to_numpy(), color.loc[c_new.index].to_numpy()), flush=True)

# (1) 피처 일치 (pool, evalp, Test)
ok = True
for nm, fr in [('pool', pool), ('evalp', evalp)]:
    X_new = M.build_features(fr, color)
    X_old = pd.concat([feats.build(fr, color, {'extra': True, 'cum': False}), rules.rule_features(fr)], axis=1)
    ok &= same(X_new, X_old, nm)
te = pd.read_csv(DATA / 'test/test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
tcol = pd.read_pickle(CACHE / 'test_color.pkl')
ok &= same(M.build_features(te, tcol), pd.concat([feats.build(te, tcol, {'extra': True, 'cum': False}), rules.rule_features(te)], axis=1), 'test')
print('  피처 수:', M.build_features(te, tcol).shape[1], flush=True)

# (2) 공통 CV rep 0 (model.py 경로)
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int)
oof = pd.DataFrame(index=evalp.index)
for f in cv.make_folds(pool, n_repeats=1):
    tr = pool.experiment_id.isin(f['train_exps']).to_numpy()
    va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    m = M.build_model().fit(Xp[tr], y[tr])
    P = m.predict_proba(Xe[va])
    oof.loc[evalp.index[va], 'pred_r0'] = np.argmax(P, 1) + 1
    for k in range(6):
        oof.loc[evalp.index[va], f'p{k+1}_r0'] = P[:, k]
oof['pred_r0'] = oof['pred_r0'].astype(int)
res = cv.evaluate(evalp, oof)
ref = pd.read_pickle(EXP / 'results/oof/rules_tab_nocum_r3.pkl')
same_pred = (oof['pred_r0'] == ref.loc[oof.index, 'pred_r0']).mean()
dp = np.abs(oof[[f'p{k}_r0' for k in range(1, 7)]].to_numpy() - ref.loc[oof.index, [f'p{k}_r0' for k in range(1, 7)]].to_numpy()).max()
print(f'  CV rep0 (model.py) {res["cv"]:.4f}, 참조 rules_tab_nocum_r3 rep0 {cv.macro_f1(evalp.label, ref.loc[evalp.index, "pred_r0"]):.4f}', flush=True)
print(f'  예측 일치율 {same_pred:.6f}, 확률 최대 차이 {dp:.2e}', flush=True)
oof.to_pickle(CACHE / 'final/verify_a_oof_r0.pkl')
print(f'피처 일치 {ok}, 총 {time.time() - t0:.0f}s')
