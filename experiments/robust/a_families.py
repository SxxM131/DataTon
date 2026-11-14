"""기본 모델(6단계 테이블 모델)을 다른 트리 계열로 바꾸거나 섞으면 나아지는가. 공통 검증 3회, 최종 규칙 체인(v9) 적용.

현재: LightGBM. 후보: XGBoost(깊이 기준), ExtraTrees(무작위 분할), RandomForest — 같은 피처 296개, 같은 fold.
각각 단독, 그리고 현재 모델과의 확률 평균. 2·3단계 재판정 모델과 규칙은 그대로 둔다.
사용: python experiments/robust/a_families.py
"""
import sys
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

C = Path.home() / 'jaehyeok' / 'cache'; NJ = 6
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int); ye = evalp.label.to_numpy(int)
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
pa = np.stack([A[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]); pb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
NAMES = ('lgb', 'xgb', 'et', 'rf'); out = C / 'a_families.npz'
if out.exists():
    Pm = dict(np.load(out))
else:
    Pm = {k: np.full((3, len(evalp), 6), np.nan) for k in NAMES}; t0 = time.time()
    for f in cv.make_folds(pool, n_repeats=3):
        r = f['rep']; tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        Xtr, Xva, ytr = Xp[tr], Xe[va], y[tr]
        Pm['lgb'][r, va] = lgb.LGBMClassifier(**{**M.LGB_PARAMS, 'n_jobs': NJ}).fit(Xtr, ytr).predict_proba(Xva)
        Pm['xgb'][r, va] = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.3, min_child_weight=5, tree_method='hist',
                                             max_bin=64, n_jobs=NJ, random_state=42, verbosity=0).fit(Xtr, ytr - 1).predict_proba(Xva)
        Ftr, Fva = Xtr.fillna(-999.0), Xva.fillna(-999.0)
        Pm['et'][r, va] = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.3, n_jobs=NJ, random_state=42).fit(Ftr, ytr).predict_proba(Fva)
        Pm['rf'][r, va] = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features=0.3, n_jobs=NJ, random_state=42).fit(Ftr, ytr).predict_proba(Fva)
        print(f"fold {r}-{f['fold']} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, **Pm)

cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])


def frames(p):
    """p: (3, n, 6) 확률 → (테이블 단독 예측, 최종 규칙 체인 적용 예측) DataFrame."""
    raw, fin = pd.DataFrame(index=evalp.index), pd.DataFrame(index=evalp.index)
    for r in range(3):
        raw[f'pred_r{r}'] = p[r].argmax(1) + 1; fin[f'pred_r{r}'] = M.apply_rules(raw[f'pred_r{r}'].to_numpy(), pb[r], evalp)
    return raw, fin


def mf(df, m=None):
    m = np.ones(len(ye), bool) if m is None else m
    return float(np.mean([f1_score(ye[m], df[c].to_numpy()[m], labels=range(1, 7), average='macro') for c in df.columns]))


cur_raw, cur_fin = frames(pa)
print(f"저장된 현재 모델 OOF와 다시 학습한 LightGBM의 예측 일치율: {np.mean([(Pm['lgb'][r].argmax(1) == pa[r].argmax(1)).mean() for r in range(3)]):.4f}")
print(f'현재 (LightGBM): 테이블 단독 {mf(cur_raw):.4f} | 규칙 체인 후 {mf(cur_fin):.4f} | Test 구성 {mf(cur_fin, TEST):.4f}')
cands = {'XGBoost 단독': Pm['xgb'], 'ExtraTrees 단독': Pm['et'], 'RandomForest 단독': Pm['rf'],
         'LightGBM + XGBoost 평균': (pa + Pm['xgb']) / 2, 'LightGBM + ExtraTrees 평균': (pa + Pm['et']) / 2, 'LightGBM + RandomForest 평균': (pa + Pm['rf']) / 2,
         'LightGBM + XGBoost + ExtraTrees 평균': (pa + Pm['xgb'] + Pm['et']) / 3, '네 모델 평균': (pa + Pm['xgb'] + Pm['et'] + Pm['rf']) / 4}
for name, p in cands.items():
    raw, fin = frames(p); c = cv.compare(evalp, fin, cur_fin, n_boot=2000); pc = np.mean([f1_score(ye, fin[k], labels=range(1, 7), average=None) for k in fin.columns], axis=0)
    print(f"{name:34s} 테이블 단독 {mf(raw):.4f} ({mf(raw) - mf(cur_raw):+.4f}) | 규칙 체인 후 {mf(fin):.4f} ({c['diff']:+.4f}, 개선 확률 {c['p_a_better']:.2f}) | Test 구성 {mf(fin, TEST):.4f} ({mf(fin, TEST) - mf(cur_fin, TEST):+.4f}) | 단계별 {np.round(pc, 3).tolist()}", flush=True)
