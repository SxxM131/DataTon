"""이미지 모델의 임베딩을 표(테이블 모델 입력)에 붙이면 나아지는가 (특징 수준 융합).

임베딩: 미세조정하지 않은 사전학습 모델의 출력 (라벨을 쓰지 않으므로 fold 누수 없음).
  resnet18  : ~/jaehyeok/cache/img/emb_resnet18_rgb.npy (512차원, 이전에 만들어 둔 것)
  dinov2    : DINOv2 ViT-S/14, 126px, CLS + 패치 평균 (768차원)  → python emb_table.py --make dinov2
붙이는 방법: 학습 fold 행으로 표준화 + PCA(k) → 현재 프레임 성분 k개 [+ 직전 관측 대비 초당 변화 k개]. 직전 관측은 그 입력에 보이는 행만 (인과적).
평가: 공통 검증 3회, 같은 LightGBM 설정, 2·3 재판정과 규칙 체인(v9) 그대로.
사용: python experiments/img/emb_table.py --make dinov2 ;  python experiments/img/emb_table.py resnet18 dinov2
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent / 'common')]
C = Path.home() / 'jaehyeok' / 'cache'
FILES = {'resnet18': C / 'img' / 'emb_resnet18_rgb.npy', 'dinov2': C / 'img' / 'emb_dinov2_s14_126.npy'}

if len(sys.argv) > 2 and sys.argv[1] == '--make':
    import timm
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    X = np.load(C / 'img' / 'pool_rgb128.npy', mmap_mode='r')
    m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=126).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            x = torch.from_numpy(np.asarray(X[i:i + 256])).permute(0, 3, 1, 2).float().div(255)
            x = F.interpolate((x - mean) / std, size=(126, 126), mode='bilinear', align_corners=False)
            t = m.forward_features(x); out.append(torch.cat([t[:, 0], t[:, m.num_prefix_tokens:].mean(1)], 1).numpy().astype(np.float16))
            if i % 5120 == 0:
                print(i, flush=True)
    np.save(FILES['dinov2'], np.concatenate(out)); print('saved', FILES['dinov2']); raise SystemExit

import lightgbm as lgb  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
import cv  # noqa: E402
import model as M  # noqa: E402

pool = cv.load_pool(Path.home() / 'data' / 'battery_data_light'); evalp = cv.to_test_like(pool)
ids = np.load(C / 'img' / 'pool_ids.npy'); assert (ids == pool.sample_id.to_numpy().astype(str)).all()
color = pd.concat([pd.read_pickle(C / f'{s}_color.pkl') for s in ('train', 'validation')])
Xp, Xe = M.build_features(pool, color), M.build_features(evalp, color)
y = pool.label.to_numpy(int); ye = evalp.label.to_numpy(int); pos_e = pool.index.get_indexer(evalp.index)
R = Path.home() / 'jaehyeok' / 'DataTon' / 'experiments' / 'results' / 'oof'
A = pd.read_pickle(R / 'rules_tab_nocum_r3.pkl').reindex(evalp.index); B = pd.read_pickle(R / 'st3_presrise_mask_r3.pkl').reindex(evalp.index)
pb = [B[[f'p{k}_r{r}' for k in range(1, 7)]].to_numpy() for r in range(3)]
cur = pd.DataFrame({f'pred_r{r}': M.apply_rules(A[f'pred_r{r}'].to_numpy(int), pb[r], evalp) for r in range(3)}, index=evalp.index)
cfg = (evalp.bat_class.astype(int).astype(str) + '/' + evalp.bat_cc.astype(str) + '/' + evalp.tr_method.astype(int).astype(str)).to_numpy()
TEST = np.isin(cfg, ['1/4.8/1', '1/6.0/1', '2/39.0/1', '2/73.0/2', '3/28.0/1', '3/280.0/2'])
mf = lambda d, m=None: float(np.mean([f1_score(ye if m is None else ye[m], d[k].to_numpy() if m is None else d[k].to_numpy()[m], labels=range(1, 7), average='macro') for k in d.columns]))  # noqa: E731
prev_p, prev_e = M.prev_index(pool.reset_index(drop=True)), M.prev_index(evalp.reset_index(drop=True))          # 그 입력에 보이는 직전 관측
tp, te = pool.relative_time_s.to_numpy(float), evalp.relative_time_s.to_numpy(float)
print(f'현재: 규칙 체인 후 {mf(cur):.4f} | Test 구성 {mf(cur, TEST):.4f}', flush=True)
FOLDS = cv.make_folds(pool, n_repeats=3)
for name in sys.argv[1:]:
    E = np.load(FILES[name]).astype(np.float32); Ee = E[pos_e]
    for k, rate in ((8, False), (16, False), (16, True), (32, True)):
        raw, fin = pd.DataFrame(index=evalp.index), pd.DataFrame(index=evalp.index)
        for r in range(3):
            p = np.zeros((len(evalp), 6))
            for f in FOLDS:
                if f['rep'] != r:
                    continue
                tr = pool.experiment_id.isin(f['train_exps']).to_numpy(); va = evalp.experiment_id.isin(f['valid_exps']).to_numpy()
                mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-6; pca = PCA(k, random_state=0).fit((E[tr][::3] - mu) / sd)   # 통계는 학습 fold에서만
                Zp, Ze = pca.transform((E - mu) / sd), pca.transform((Ee - mu) / sd); cols = [f'emb{j}' for j in range(k)]
                Fp, Fe = pd.DataFrame(Zp, index=pool.index, columns=cols), pd.DataFrame(Ze, index=evalp.index, columns=cols)
                if rate:
                    for Fx, Z, pv, t in ((Fp, Zp, prev_p, tp), (Fe, Ze, prev_e, te)):
                        dt = np.maximum(t - t[pv], 1e-6)[:, None]; d = np.where((pv != np.arange(len(pv)))[:, None], (Z - Z[pv]) / dt, 0.0)
                        for j in range(k):
                            Fx[f'emb{j}_rate'] = d[:, j]
                m = lgb.LGBMClassifier(**{**M.LGB_PARAMS, 'n_jobs': 6}).fit(pd.concat([Xp[tr], Fp[tr]], axis=1), y[tr])
                p[va] = m.predict_proba(pd.concat([Xe[va], Fe[va]], axis=1))
            raw[f'pred_r{r}'] = p.argmax(1) + 1; fin[f'pred_r{r}'] = M.apply_rules(raw[f'pred_r{r}'].to_numpy(), pb[r], evalp)
        c = cv.compare(evalp, fin, cur, n_boot=2000); pc = np.mean([f1_score(ye, fin[q], labels=range(1, 7), average=None) for q in fin.columns], axis=0)
        print(f"{name} 임베딩 PCA {k}개{' + 변화율' if rate else ''}: 테이블 단독 {mf(raw):.4f} | 규칙 체인 후 {mf(fin):.4f} ({c['diff']:+.4f}, 개선 확률 {c['p_a_better']:.2f}) | Test 구성 {mf(fin, TEST):.4f} ({mf(fin, TEST) - mf(cur, TEST):+.4f}) | 단계별 {np.round(pc, 3).tolist()}", flush=True)
