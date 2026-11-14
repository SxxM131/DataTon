"""기본 모델을 여러 트리 계열(LightGBM, XGBoost, CatBoost, ExtraTrees, RandomForest)의 블렌드로 바꾸면 나아지는가.

a_families.py / a_families_cb.py가 저장한 OOF 확률(공통 검증 3회)을 쓴다. 2·3단계 재판정과 규칙 체인(v9)은 그대로.
  (1) 모든 조합(31개)의 동일 가중 산술 평균·기하 평균
  (2) LightGBM 비중을 바꾼 가중 평균
  (3) 조합을 fold 밖에서 고르기: 각 fold의 조합을 나머지 4개 fold의 점수로 골라 그 fold에 적용 (여러 조합 중 최고를 고르는 낙관을 걷어낸 값)
사용: python experiments/robust/a_blend.py
"""
import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
import cv  # noqa: E402
import model as M  # noqa: E402

C = Path.home() / 'jaehyeok' / 'cache'
pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool); ye = evalp.label.to_numpy(int)
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
pb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
Pm = np.load(C / 'a_families.npz')
P = {'LGB': np.stack([A[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]), 'XGB': Pm['xgb'], 'CB': np.load(C / 'a_families_cb.npz')['cb'], 'ET': Pm['et'], 'RF': Pm['rf']}
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])
fold = np.full((3, len(evalp)), -1)
for f in cv.make_folds(pool, n_repeats=3):
    fold[f['rep'], evalp.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']


def chain(p):
    """p: (3, n, 6) → 규칙 체인 적용 예측 (3, n)."""
    return np.stack([M.apply_rules(p[r].argmax(1) + 1, pb[r], evalp) for r in range(3)])


def mf(pred, m=None):
    m = np.ones(len(ye), bool) if m is None else m
    return float(np.mean([f1_score(ye[m], pred[r][m], labels=range(1, 7), average='macro') for r in range(3)]))


def df(pred):
    return pd.DataFrame({f'pred_r{r}': pred[r] for r in range(3)}, index=evalp.index)


cur = chain(P['LGB']); cur_df = df(cur); base = mf(cur); base_t = mf(cur, TEST)
print(f'현재 (LightGBM 단독): 규칙 체인 후 {base:.4f} | Test 구성 {base_t:.4f}\n')
cands = {}
for k in range(1, 6):
    for combo in itertools.combinations(P, k):
        ps = [P[c] for c in combo]
        cands['+'.join(combo) + ' 산술'] = np.mean(ps, axis=0)
        if k > 1:
            g = np.exp(np.mean([np.log(np.clip(p, 1e-9, 1)) for p in ps], axis=0)); cands['+'.join(combo) + ' 기하'] = g / g.sum(-1, keepdims=True)
for w in (0.9, 0.8, 0.7, 0.6):
    cands[f'LGB {w} + XGB {1 - w:.1f}'] = w * P['LGB'] + (1 - w) * P['XGB']
    cands[f'LGB {w} + CB {1 - w:.1f}'] = w * P['LGB'] + (1 - w) * P['CB']
    cands[f'LGB {w} + (XGB+CB) {1 - w:.1f}'] = w * P['LGB'] + (1 - w) * (P['XGB'] + P['CB']) / 2
    cands[f'LGB {w} + (XGB+CB+ET+RF) {1 - w:.1f}'] = w * P['LGB'] + (1 - w) * (P['XGB'] + P['CB'] + P['ET'] + P['RF']) / 4
preds = {n: chain(p) for n, p in cands.items()}
rows = []
for n, pr in preds.items():
    c = cv.compare(evalp, df(pr), cur_df, n_boot=2000)
    rows.append({'구성': n, '규칙 체인 후': round(mf(pr), 4), '현재 대비': round(c['diff'], 4), '개선 확률': round(c['p_a_better'], 2), 'Test 구성': round(mf(pr, TEST), 4), 'Test 구성 대비': round(mf(pr, TEST) - base_t, 4)})
T = pd.DataFrame(rows).sort_values('규칙 체인 후', ascending=False)
print(f'후보 {len(T)}개 (단독 5 + 블렌드 {len(T) - 5}). 현재보다 높은 후보 {int((T["현재 대비"] > 0).sum())}개, 개선 확률 0.9 이상 {int((T["개선 확률"] >= 0.9).sum())}개')
print('--- 상위 12개 ---'); print(T.head(12).to_string(index=False))
print('--- 단독 모델 ---'); print(T[T['구성'].isin([f'{k} 산술' for k in P])].to_string(index=False))

# (3) 조합을 fold 밖에서 고르기
names = list(preds); nested = np.zeros((3, len(ye)), int); picks = []
for r in range(3):
    for k in range(5):
        tr, va = (fold[r] != k) & (fold[r] >= 0), fold[r] == k
        best = max(names, key=lambda n: f1_score(ye[tr], preds[n][r][tr], labels=range(1, 7), average='macro')); picks.append(best)
        nested[r, va] = preds[best][r][va]
c = cv.compare(evalp, df(nested), cur_df, n_boot=2000)
print(f'\n조합을 fold 밖에서 골랐을 때: {mf(nested):.4f} (현재 대비 {c["diff"]:+.4f}, 개선 확률 {c["p_a_better"]:.2f}) | Test 구성 {mf(nested, TEST):.4f} ({mf(nested, TEST) - base_t:+.4f})')
print('fold별로 뽑힌 조합:', pd.Series(picks).value_counts().to_dict())
T.to_csv(C / 'a_blend_table.csv', index=False)
