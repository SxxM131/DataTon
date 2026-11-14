"""시드 평균(같은 설정의 LightGBM을 시드만 바꿔 여러 개 학습해 확률 평균)의 효과. 공통 검증 3회.

목적: Private 방어 — 한 번 학습한 모델의 우연(행 샘플링·피처 샘플링 시드)에 예측이 좌우되는 정도를 줄인다.
기본 모델(296 피처)과 2·3단계 재판정 모델(+압력 상승 피처) 모두 시드 k개 평균, 이후 최종 규칙 보정(apply_rules) 적용.
"""
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

SEEDS = [42, 1, 2, 3, 4]
C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int)
PA = np.zeros((3, len(SEEDS), len(evalp), 6), np.float32); PB = np.zeros_like(PA)
for f in cv.make_folds(pool, n_repeats=3):
    r = f['rep']; tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    Xb_tr = pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1)
    Xb_va = pd.concat([Xe[va], M.pressure_features(evalp[va], tab)], axis=1)
    for i, s in enumerate(SEEDS):
        P = {**M.LGB_PARAMS, 'n_jobs': 6, 'random_state': s}
        PA[r, i, va] = lgb.LGBMClassifier(**P).fit(Xp[tr], y[tr]).predict_proba(Xe[va])
        PB[r, i, va] = lgb.LGBMClassifier(**P).fit(Xb_tr, y[tr]).predict_proba(Xb_va)
    print('fold', r, f['fold'], flush=True)
    if f['fold'] == 4:
        np.savez_compressed(C / 'robust_seed_bag.npz', PA=PA, PB=PB, index=evalp.index.to_numpy())
        for k in (1, 3, 5):
            o = pd.DataFrame({f'pred_r{q}': M.apply_rules(M.to_label(PA[q, :k].mean(0)), PB[q, :k].mean(0), evalp) for q in range(r + 1)}, index=evalp.index)
            rr = cv.evaluate(evalp, o, n_boot=100)
            print(f"  [rep 0~{r}] 시드 {k}개 평균: CV {rr['cv']:.4f} {np.round(rr['per_class'], 3).tolist()} | long {rr['long_exps']:.3f} short {rr['short_exps']:.3f}", flush=True)
base = pd.DataFrame({f'pred_r{q}': M.apply_rules(M.to_label(PA[q, 0]), PB[q, 0], evalp) for q in range(3)}, index=evalp.index)
for k in (3, 5):
    o = pd.DataFrame({f'pred_r{q}': M.apply_rules(M.to_label(PA[q, :k].mean(0)), PB[q, :k].mean(0), evalp) for q in range(3)}, index=evalp.index)
    c = cv.compare(evalp, o, base, n_boot=3000); print(f"시드 {k}개 평균 − 시드 1개: {c['diff']:+.4f} [{c['ci90'][0]:+.3f},{c['ci90'][1]:+.3f}] p {c['p_a_better']:.2f}", flush=True)
# 시드 하나씩의 점수 편차 (우연의 크기)
one = [cv.evaluate(evalp, pd.DataFrame({f'pred_r{q}': M.apply_rules(M.to_label(PA[q, i]), PB[q, i], evalp) for q in range(3)}, index=evalp.index), n_boot=10)['cv'] for i in range(len(SEEDS))]
print('시드별 단일 모델 CV:', np.round(one, 4).tolist(), '| 표준편차', round(float(np.std(one)), 4))
