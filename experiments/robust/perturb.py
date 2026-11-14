"""견고성 검증: 검증 실험의 압력 센서 값에 상수를 더하거나 곱한 뒤 채점 (공통 검증 rep 0, 모델은 원래 데이터로 학습).

라벨 규칙은 압력을 실험별 min-max 정규화해서 쓰므로, 압력에 상수를 더하거나 곱해도 정답은 그대로다.
→ 점수가 떨어지면 모델이 절대 압력값(센서 기준값·스케일)에 의존한다는 뜻. Test에는 초기 압력이 학습 범위 밖인 실험이 있다.
--center: 테이블 피처를 만들기 전에 압력에서 실험 첫 관측값을 뺀다 (기준값 불변). --no_p0: 초기 압력 대비 q 피처 제거.
"""
import argparse, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('--center', action='store_true'); ap.add_argument('--no_p0', action='store_true'); ap.add_argument('--tag', default='base')
args = ap.parse_args()
C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
y = pool.label.to_numpy(int)
P0COLS = [c for c in M.Q_COLS if 'p0' in c]


def center(fr):
    """압력에서 그 실험의 첫 관측값(입력 행 중 가장 이른 시각)을 뺀다 — 과거 정보만 사용."""
    first = fr.sort_values('relative_time_s').groupby('experiment_id').meas_p.first()
    return fr.assign(meas_p=fr.meas_p - fr.experiment_id.map(first).to_numpy())


def feats(fr):
    return M.build_features(center(fr) if args.center else fr, color)


def qfeats(fr, table, own=None):
    q = M.pressure_features(fr, table, own_range=own)
    return q.drop(columns=P0COLS) if args.no_p0 else q


PERT = [('원본', 1.0, 0.0), ('+10', 1.0, 10.0), ('+30', 1.0, 30.0), ('−(초기값)', None, None), ('×0.3', 0.3, 0.0), ('×3', 3.0, 0.0), ('×3 +10', 3.0, 10.0)]
Xp = feats(pool)
folds = [(pool.experiment_id.isin(f['train_exps']).to_numpy(), evalp.experiment_id.isin(f['valid_exps']).to_numpy()) for f in cv.make_folds(pool, n_repeats=1)]
PR = {**M.LGB_PARAMS, 'n_jobs': 6}
models = []
for tr, va in folds:
    er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
    a = lgb.LGBMClassifier(**PR).fit(Xp[tr], y[tr])
    b = lgb.LGBMClassifier(**PR).fit(pd.concat([Xp[tr], qfeats(pool[tr], tab, M.loo_ranges(er))], axis=1), y[tr])
    models.append((a, b, tab)); print('학습', len(models), flush=True)
ye = evalp.label.to_numpy(int)
from sklearn.metrics import f1_score
res = {}
for name, s, c in PERT:
    ev = center(evalp) if s is None else evalp.assign(meas_p=evalp.meas_p * s + c)
    Xe = feats(ev)
    pa = np.zeros(len(ev), int); pb = np.zeros(len(ev), int)
    for (tr, va), (a, b, tab) in zip(folds, models):
        pa[va] = a.predict(Xe[va])
        p23 = b.predict_proba(pd.concat([Xe[va], qfeats(ev[va], tab)], axis=1))
        pb[va] = M.redecide_23(pa[va], p23)
    res[name] = pb
    f = lambda p: np.round(f1_score(ye, p, labels=range(1, 7), average=None), 3).tolist()  # noqa: E731
    print(f"{name:10s} 기본 모델 {cv.macro_f1(ye, pa):.4f} {f(pa)} | + 2·3 재판정 {cv.macro_f1(ye, pb):.4f} {f(pb)}", flush=True)
pd.DataFrame(res, index=evalp.index).to_pickle(C / f'robust_perturb_{args.tag}.pkl')
