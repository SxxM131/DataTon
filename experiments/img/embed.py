"""고정(frozen) ImageNet 백본 임베딩 → (a) 이미지 단독 분류, (b) PCA(학습 fold에서만 fit) + 센서 GBDT.

백본: timm resnet18.tv_in1k (torchvision ImageNet-1k, BSD-3-Clause). 입력: RGB 128 또는 의사온도 3채널 복제.
시간 문맥: 임베딩 차이(현재-첫 프레임, 현재-30초 전)를 pool/evalp 문맥에서 각각 계산.
"""
import sys
import numpy as np
import pandas as pd
import torch
import timm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import imglib as L
from imglib import cv

torch.set_num_threads(4)
pool, evalp = L.load_all()
N = len(pool)
y = pool.label.astype(int).to_numpy()
is_eval = np.zeros(N, bool); is_eval[pool.index.get_indexer(evalp.index)] = True
ctx_pool = L.context_index(pool, np.ones(N, bool)); ctx_eval = L.context_index(pool, is_eval)
ev_rows = np.where(is_eval)[0]


@torch.no_grad()
def embed(kind):
    f = L.ICACHE / f'emb_resnet18_{kind}.npy'
    if f.exists():
        return np.load(f)
    m = timm.create_model('resnet18.tv_in1k', pretrained=True, num_classes=0).cuda().eval()
    X = L.load_images(pool, kind, 128)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    out = []
    for i in range(0, N, 1024):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + 1024])).cuda().float() / 255
        x = x.permute(0, 3, 1, 2) if kind == 'rgb' else x[:, None].expand(-1, 3, -1, -1)
        out.append(m((x - mean) / std).cpu().numpy())
    out = np.concatenate(out).astype(np.float32)
    np.save(f, out)
    return out


Xs_fit, Xs_eval = L.sensor_features(pool, evalp)
Xs_fit, Xs_eval = Xs_fit.to_numpy(np.float32), Xs_eval.to_numpy(np.float32)
oofs = {}
for kind in ('rgb', 'pt'):
    E = embed(kind)

    def feats(ctx, rows, temporal):
        e = E[rows]
        return np.hstack([e, e - E[ctx['first'][rows]], e - E[ctx['lag30'][rows]]]) if temporal else e

    for temporal in (False, True):
        A_fit, A_eval = feats(ctx_pool, np.arange(N), temporal), feats(ctx_eval, ev_rows, temporal)
        tagn = f"img_emb_{kind}{'_temporal' if temporal else ''}"

        def lr_only(tr, va, f):
            m = make_pipeline(StandardScaler(), PCA(64, random_state=0), LogisticRegression(C=0.5, max_iter=300, class_weight='balanced'))
            sub = np.where(tr)[0][::3]                      # 5초 간격 행은 중복이 많아 1/3만 사용 (속도)
            return m.fit(A_fit[sub], y[sub]).predict(A_eval[va])

        def fused(tr, va, f):
            p = make_pipeline(StandardScaler(), PCA(16, random_state=0)).fit(A_fit[np.where(tr)[0][::3]])
            m = L.lgbm().fit(np.hstack([Xs_fit[tr], p.transform(A_fit[tr])]), y[tr])
            return m.predict(np.hstack([Xs_eval[va], p.transform(A_eval[va])]))

        _, oofs[tagn + '_lr'] = L.run_cv(tagn + '_lr', pool, evalp, lr_only)
        _, oofs[tagn + '_gbdt_fused'] = L.run_cv(tagn + '_pca16_sensor_gbdt', pool, evalp, fused)
import os
p = L.HERE.parent / 'results/oof/img_tab_sensor_color.pkl'
if p.exists():
    base = pd.read_pickle(p)
    for k, v in oofs.items():
        if 'fused' in k:
            L.cmp(evalp, v, base, k, 'lgbm sensor+color')
