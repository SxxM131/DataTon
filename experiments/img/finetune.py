"""fold별 CNN 미세조정 → 이미지 6-클래스 확률 (OOF, 스태킹용).

입력 (mode)
  single   : 현재 프레임 RGB (3ch)
  temporal : 현재 RGB + 약 30초 전 프레임 RGB + 실험 첫 프레임 RGB (9ch). 모두 같은 실험의 현재/과거 프레임만 사용.
             학습 행은 pool(5초 간격) 문맥에서 lag 30/35/40초 중 무작위(Test의 10/15초 간격 대응),
             평가 행은 evalp(Test처럼 다운샘플) 문맥에서 't-30초 이하의 마지막 관측'.
모델: timm resnet18.tv_in1k (torchvision ImageNet-1k 가중치, BSD-3-Clause) — 대회 전 공개 범용 가중치.
증강: 좌우 반전 + 약한 affine(스케일/이동). 색 변형 없음(팔레트 의미 보존).

학습하는 모델 (누수 없는 중첩 스태킹용)
  out k     : fold k 제외 4개 fold로 학습 → fold k 예측 (pool 문맥 + evalp 문맥)
  pair k,j  : fold k, j 제외 3개 fold로 학습 → fold k, j 예측.  outer fold k의 GBDT 학습 행(fold j)에는 pair(k,j) 확률을 쓴다.
  official  : 제공 Train 78 → Validation 19
출력: ~/jaehyeok/cache/img/cnn_{tag}.npz
"""
import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F
import timm

import imglib as L
from imglib import cv

torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='temporal')
ap.add_argument('--tag', default=None)
ap.add_argument('--arch', default='resnet18.tv_in1k')
ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--epoch_size', type=int, default=24000)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--S', type=int, default=128)
ap.add_argument('--no_pairs', action='store_true')
ap.add_argument('--smoke', action='store_true')
args = ap.parse_args()
tag = args.tag or args.mode
dev = 'cuda'

pool, evalp = L.load_all()
N = len(pool)
y = torch.tensor(pool.label.to_numpy() - 1, device=dev)
is_eval = np.zeros(N, bool); is_eval[pool.index.get_indexer(evalp.index)] = True
ctx_pool = L.context_index(pool, np.ones(N, bool), lags=(30, 35, 40))
ctx_eval = L.context_index(pool, is_eval, lags=(30,))
X = torch.from_numpy(np.load(L.ICACHE / f'pool_rgb{args.S}.npy')).to(dev)          # (N,S,S,3) uint8
assert (np.load(L.ICACHE / 'pool_ids.npy') == pool.sample_id.to_numpy().astype(str)).all()
C = {'single': 3, 'temporal': 9}[args.mode]
MEAN = torch.tensor([0.485, 0.456, 0.406] * (C // 3), device=dev).view(1, C, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225] * (C // 3), device=dev).view(1, C, 1, 1)
T = lambda a: torch.as_tensor(a, device=dev)
P_first, P_lags = T(ctx_pool['first']), torch.stack([T(ctx_pool[f'lag{w}']) for w in (30, 35, 40)])
E_first, E_lag = T(ctx_eval['first']), T(ctx_eval['lag30'])

folds = cv.make_folds(pool, n_repeats=1)
fold_of = np.full(N, -1)
for f in folds:
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
n_exp = pool.groupby('experiment_id')['label'].transform('size').to_numpy()
base_w = np.minimum(1.0, cv.CAP / n_exp)                       # 긴 실험이 행 수로 지배하지 않게 (Test의 500행 상한과 같은 비중)


def batch(idx, ctx, train):
    """idx: pool 행 위치 텐서. ctx='pool'|'eval'."""
    if C == 3:
        x = X[idx]
    else:
        if ctx == 'pool':
            lag = P_lags[torch.randint(0, 3, idx.shape, device=dev) if train else torch.zeros_like(idx), idx]
            first = P_first[idx]
        else:
            lag, first = E_lag[idx], E_first[idx]
        x = torch.cat([X[idx], X[lag], X[first]], -1)
    x = x.permute(0, 3, 1, 2).float().div_(255)
    if train:
        B = len(idx)
        s = 1 + (torch.rand(B, device=dev) - .5) * .3
        flip = (torch.rand(B, device=dev) < .5).float() * 2 - 1
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0] = s * flip; th[:, 1, 1] = s * (1 + (torch.rand(B, device=dev) - .5) * .1)
        th[:, :, 2] = (torch.rand(B, 2, device=dev) - .5) * .2
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode='bilinear', padding_mode='border', align_corners=False)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def train_model(train_mask, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = timm.create_model(args.arch, pretrained=True, num_classes=6, in_chans=C, drop_rate=0.2).to(dev).to(memory_format=torch.channels_last)
    idx_all = np.where(train_mask)[0]
    yy = pool.label.to_numpy()[idx_all]
    cw = (1.0 / np.bincount(yy, minlength=7)[1:].clip(1)) ** 0.5
    w = torch.tensor(base_w[idx_all] * cw[yy - 1], device=dev, dtype=torch.float)
    idx_all = T(idx_all)
    steps = args.epochs * (args.epoch_size // args.bs)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    m.train()
    g = torch.Generator(device=dev); g.manual_seed(seed)
    tot = 0.0
    for it in range(steps):
        idx = idx_all[torch.multinomial(w, args.bs, replacement=True, generator=g)]
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(m(batch(idx, 'pool', True)), y[idx], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item() if it % 20 == 0 else 0
    return m.eval()


@torch.no_grad()
def predict(m, rows, ctx):
    out = np.full((N, 6), np.nan, dtype=np.float32)
    rows_t = T(rows)
    for i in range(0, len(rows), 512):
        idx = rows_t[i:i + 512]
        out[rows[i:i + 512]] = F.softmax(m(batch(idx, ctx, False)).float(), 1).cpu().numpy()
    return out


def run(name, hold_mask, seed):
    t0 = time.time()
    m = train_model(~hold_mask, seed)
    rows = np.where(hold_mask)[0]
    pp = predict(m, rows, 'pool')
    pe = predict(m, rows[is_eval[rows]], 'eval') if C == 9 else np.where(is_eval[:, None], pp, np.nan)
    ev = hold_mask & is_eval
    f1 = cv.macro_f1(pool.label.to_numpy()[ev], pe[ev].argmax(1) + 1)
    print(f'{name}: held-out evalp rows {ev.sum()} image-only macroF1 {f1:.4f} ({time.time() - t0:.0f}s)', flush=True)
    return pp, pe, m


res = {}
if args.smoke:
    args.epochs = 1
    run('smoke', fold_of == 0, 0); raise SystemExit
for k in range(5):
    pp, pe, m = run(f'out{k}', fold_of == k, 100 + k)
    res[f'out{k}_pool'], res[f'out{k}_eval'] = pp, pe
    torch.save({n: v.half() for n, v in m.state_dict().items()}, L.ICACHE / f'cnn_{tag}_out{k}.pt')
    np.savez(L.ICACHE / f'cnn_{tag}.npz', **res)
is_off_val = (pool.origin == 'validation').to_numpy()
pp, pe, _ = run('official', is_off_val, 200)
res['official_pool'], res['official_eval'] = pp, pe
np.savez(L.ICACHE / f'cnn_{tag}.npz', **res)
if not args.no_pairs:
    for k, j in itertools.combinations(range(5), 2):
        pp, pe, _ = run(f'pair{k}{j}', (fold_of == k) | (fold_of == j), 300 + 10 * k + j)
        res[f'pair{k}{j}_pool'], res[f'pair{k}{j}_eval'] = pp, pe
        np.savez(L.ICACHE / f'cnn_{tag}.npz', **res)
print('done', flush=True)
