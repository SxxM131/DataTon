"""기본 모델을 CatBoost로 바꾸거나 섞으면 나아지는가 (a_families.py의 추가분). 같은 피처 296개, 같은 fold, 최종 규칙 체인(v9).

CatBoost: 대칭 트리, MultiClass, 500 iterations, learning_rate 0.1, depth 6, rsm 0.3, CPU.
사용: python experiments/robust/a_families_cb.py   (a_families.py를 먼저 실행해 ~/jaehyeok/cache/a_families.npz가 있어야 섞기 비교가 나온다)
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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
out = C / 'a_families_cb.npz'
if out.exists():
    pcb = np.load(out)['cb']
else:
    pcb = np.full((3, len(evalp), 6), np.nan); t0 = time.time()
    for f in cv.make_folds(pool, n_repeats=3):
        r = f['rep']; tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
        m = CatBoostClassifier(loss_function='MultiClass', iterations=500, learning_rate=0.1, depth=6, rsm=0.3, random_seed=42, thread_count=NJ, verbose=0, allow_writing_files=False)
        pcb[r, va] = m.fit(Xp[tr], y[tr]).predict_proba(Xe[va])
        print(f"fold {r}-{f['fold']} {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, cb=pcb)

cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])


def frames(p):
    raw, fin = pd.DataFrame(index=evalp.index), pd.DataFrame(index=evalp.index)
    for r in range(3):
        raw[f'pred_r{r}'] = p[r].argmax(1) + 1; fin[f'pred_r{r}'] = M.apply_rules(raw[f'pred_r{r}'].to_numpy(), pb[r], evalp)
    return raw, fin


def mf(df, m=None):
    m = np.ones(len(ye), bool) if m is None else m
    return float(np.mean([f1_score(ye[m], df[c].to_numpy()[m], labels=range(1, 7), average='macro') for c in df.columns]))


cur_raw, cur_fin = frames(pa)
print(f'현재 (LightGBM): 테이블 단독 {mf(cur_raw):.4f} | 규칙 체인 후 {mf(cur_fin):.4f} | Test 구성 {mf(cur_fin, TEST):.4f}')
cands = {'CatBoost 단독': pcb, 'LightGBM + CatBoost 평균': (pa + pcb) / 2}
if (C / 'a_families.npz').exists():
    Pm = np.load(C / 'a_families.npz')
    cands['LightGBM + XGBoost + CatBoost 평균'] = (pa + Pm['xgb'] + pcb) / 3
    cands['LightGBM + CatBoost + ExtraTrees 평균'] = (pa + pcb + Pm['et']) / 3
    cands['다섯 모델 평균'] = (pa + Pm['xgb'] + Pm['et'] + Pm['rf'] + pcb) / 5
for name, p in cands.items():
    raw, fin = frames(p); c = cv.compare(evalp, fin, cur_fin, n_boot=2000); pc = np.mean([f1_score(ye, fin[k], labels=range(1, 7), average=None) for k in fin.columns], axis=0)
    print(f"{name:34s} 테이블 단독 {mf(raw):.4f} ({mf(raw) - mf(cur_raw):+.4f}) | 규칙 체인 후 {mf(fin):.4f} ({c['diff']:+.4f}, 개선 확률 {c['p_a_better']:.2f}) | Test 구성 {mf(fin, TEST):.4f} ({mf(fin, TEST) - mf(cur_fin, TEST):+.4f}) | 단계별 {np.round(pc, 3).tolist()}", flush=True)
