"""견고성 검증: 배터리 구성 하나를 통째로 빼고 학습 → 그 구성의 실험으로 평가 (LOCO).

목적: 교차검증(같은 구성의 다른 실험으로 평가)은 '구성별 평소 단계 길이'를 외운 모델을 걸러내지 못한다.
      구성을 통째로 빼면 단계 길이·수준이 학습과 다른 실험에서의 성능을 볼 수 있다. 라벨 있는 97개 실험만 사용.
--mode cv 는 같은 변형을 공통 검증(rep 0)으로 평가해 평상시 손실을 본다.
"""
import argparse, sys, importlib.util
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='loco'); ap.add_argument('--variants', default='A,noT,cap60'); ap.add_argument('--lr', action='store_true')
ap.add_argument('--baseline_py', default=''); ap.add_argument('--tag', default='')
args = ap.parse_args()
C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int); ye = evalp.label.to_numpy(int)
META = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
SINCE = [c for c in Xp.columns if 'since' in c]


def variant(X, name):
    X = X.copy()
    for part in name.split('+'):
        if part == 'A':
            pass
        elif part == 'noT':
            X = X.drop(columns=['relative_time_s'])
        elif part.startswith('cap'):                       # 경과 시간 피처를 N초에서 자른다 (이벤트 발생 여부·직후만 알게)
            n = float(part[3:])
            X = X.drop(columns=['relative_time_s'], errors='ignore')
            X[SINCE] = X[SINCE].clip(upper=n)
        elif part == 'noMeta':
            X = X.drop(columns=META)
        else:
            raise ValueError(part)
    return X


key = lambda f: (f.bat_class.astype(int).astype(str) + '/' + f.bat_cc.astype(str) + '/' + f.tr_method.astype(int).astype(str))  # noqa: E731
kp, ke = key(pool).to_numpy(), key(evalp).to_numpy()
if args.mode == 'loco':
    folds = [(kp != c, ke == c, c) for c in sorted(set(kp))]
else:
    folds = [(pool.experiment_id.isin(f['train_exps']).to_numpy(), evalp.experiment_id.isin(f['valid_exps']).to_numpy(), f['fold']) for f in cv.make_folds(pool, n_repeats=1)]
names = args.variants.split(',')
pred = {n: np.zeros(len(evalp), int) for n in names}
if args.lr:
    spec = importlib.util.spec_from_file_location('baseline_model', args.baseline_py); BL = importlib.util.module_from_spec(spec); spec.loader.exec_module(BL)
    gray = pd.concat([pd.read_pickle(C / f'{s}_gray.pkl') for s in ('train', 'validation')])
    XLp, XLe = BL.build_features(pool, gray), BL.build_features(evalp, gray)
    pred['LR'] = np.zeros(len(evalp), int)
P = {**M.LGB_PARAMS, 'n_jobs': 6}
for tr, va, c in folds:
    for n in names:
        pred[n][va] = lgb.LGBMClassifier(**P).fit(variant(Xp[tr], n), y[tr]).predict(variant(Xe[va], n))
    if args.lr:
        pred['LR'][va] = BL.build_model(None).fit(XLp[tr], y[tr]).predict(XLe[va])
    print('완료', c, {n: round(cv.macro_f1(ye[va], p[va]), 3) for n, p in pred.items()}, flush=True)
from sklearn.metrics import f1_score
print(f'\n== {args.mode.upper()} 전체 (Macro F1, 클래스별) ==')
for n, p in pred.items():
    print(f"{n:14s} {cv.macro_f1(ye, p):.4f} {np.round(f1_score(ye, p, labels=range(1, 7), average=None, zero_division=0), 3).tolist()}")
pd.DataFrame(pred, index=evalp.index).to_pickle(C / f'robust_{args.mode}{args.tag}.pkl')
