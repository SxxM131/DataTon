"""기본 모델(LightGBM)의 하이퍼파라미터를 Optuna로 탐색하면 나아지는가.

1단계: 공통 검증 반복 0의 5-fold에서 탐색 (목표 = 최종 규칙 체인까지 적용한 Macro F1). 현재 설정을 첫 시도로 넣는다.
2단계: 가장 좋은 설정을 탐색에 쓰지 않은 반복 1·2의 fold로 다시 학습·평가해 현재 설정과 짝지어 비교한다.
       (같은 97개 실험을 다르게 나눈 것이라 완전히 독립은 아니다 — 여기서도 이득이 남을 때만 중첩 검증으로 확인할 가치가 있다.)
사용: PYTHONPATH=~/jaehyeok/cache/pylib python experiments/robust/hpo_lgb.py [시도 횟수]
"""
import sys
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 40; NJ = 6
C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int); ye = evalp.label.to_numpy(int)
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
pa = [A[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]; pb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
FOLDS = cv.make_folds(pool, n_repeats=3)
MASKS = [(f['rep'], pool.experiment_id.isin(f['train_exps']).to_numpy(), evalp.experiment_id.isin(f['valid_exps']).to_numpy()) for f in FOLDS]


def oof(params, rep):
    p = np.full((len(evalp), 6), np.nan)
    for r, tr, va in MASKS:
        if r == rep:
            p[va] = lgb.LGBMClassifier(**{**M.LGB_PARAMS, **params, 'n_jobs': NJ}).fit(Xp[tr], y[tr]).predict_proba(Xe[va])
    return p


def chain(p, rep):
    return M.apply_rules(p.argmax(1) + 1, pb[rep], evalp)


CUR = {k: M.LGB_PARAMS[k] for k in ('learning_rate', 'n_estimators', 'num_leaves', 'min_child_samples', 'subsample', 'colsample_bytree', 'reg_lambda', 'max_bin')}
cur0 = cv.macro_f1(ye, chain(pa[0], 0)); print(f'현재 설정 {CUR}\n반복 0, 규칙 체인 후 {cur0:.4f} (테이블 단독 {cv.macro_f1(ye, pa[0].argmax(1) + 1):.4f})', flush=True)
T0 = time.time()


def objective(trial):
    params = dict(learning_rate=trial.suggest_float('learning_rate', 0.03, 0.2, log=True), n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
                  num_leaves=trial.suggest_int('num_leaves', 7, 63), min_child_samples=trial.suggest_int('min_child_samples', 10, 200, log=True),
                  subsample=trial.suggest_float('subsample', 0.5, 1.0), colsample_bytree=trial.suggest_float('colsample_bytree', 0.15, 0.8),
                  reg_lambda=trial.suggest_float('reg_lambda', 0.1, 20.0, log=True), max_bin=trial.suggest_categorical('max_bin', [31, 63, 127, 255]))
    cw = trial.suggest_categorical('class_weight', ['none', 'balanced'])
    if cw == 'balanced':
        params['class_weight'] = 'balanced'
    p = oof(params, 0); s = cv.macro_f1(ye, chain(p, 0)); trial.set_user_attr('table_only', cv.macro_f1(ye, p.argmax(1) + 1))
    print(f"시도 {trial.number:2d} 규칙 체인 후 {s:.4f} (현재 대비 {s - cur0:+.4f}) 테이블 단독 {trial.user_attrs['table_only']:.4f} | {time.time() - T0:.0f}s | {trial.params}", flush=True)
    return s


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.enqueue_trial({**CUR, 'class_weight': 'none'})
study.optimize(objective, n_trials=N_TRIALS)
df = study.trials_dataframe().sort_values('value', ascending=False); df.to_csv(C / 'hpo_lgb_trials.csv', index=False)
vals = df.value.to_numpy(); print(f'\n시도 {len(vals)}회: 최고 {vals[0]:.4f}, 상위 5개 {np.round(vals[:5], 4).tolist()}, 중앙값 {np.median(vals):.4f}, 현재 설정 {cur0:.4f} (현재보다 높은 시도 {int((vals > cur0).sum())}개)')

O = lambda cols: pd.DataFrame(cols, index=evalp.index)  # noqa: E731
cur_df = O({f'pred_r{r}': chain(pa[r], r) for r in (1, 2)})
print('\n=== 상위 3개 설정을 반복 1·2(탐색에 쓰지 않은 분할)에서 다시 평가 ===')
for rank in range(3):
    t = study.trials[int(df.number.iloc[rank])]; params = dict(t.params); cw = params.pop('class_weight')
    if cw == 'balanced':
        params['class_weight'] = 'balanced'
    new_df = O({f'pred_r{r}': chain(oof(params, r), r) for r in (1, 2)}); c = cv.compare(evalp, new_df, cur_df, n_boot=2000)
    f = lambda d: float(np.mean([cv.macro_f1(ye, d[k].to_numpy()) for k in d.columns]))  # noqa: E731
    print(f"{rank + 1}위 (반복 0에서 {t.value:.4f}, 현재 대비 {t.value - cur0:+.4f}) → 반복 1·2: {f(new_df):.4f} vs 현재 {f(cur_df):.4f} ({c['diff']:+.4f}, 개선 확률 {c['p_a_better']:.2f}) | {t.params}", flush=True)
