"""[라운드 2-3] 후기 단계용 CNN: 현재 프레임 + 직전 프레임 + 차이(9ch) (+ 선택: 센서 절대값·변화량 멀티모달 헤드).

- 직전 프레임은 '시간상 바로 앞의 보이는 관측'. 학습(pool, 5초 간격)에서는 5/10/15초 전 중 무작위 → Test의 10/15초 간격에도 대응.
  평가(evalp 문맥)에서는 보이는 행 중 바로 앞 행. 실험 첫 행은 자기 자신(차이 0).
- 센서 입력(--sens): log1p(t1), log1p(t2), v, p, 직전 관측 대비 Δt1/10, Δt2/10, Δv, Δp, dt/10 → 학습 fold 평균/표준편차로 표준화(결측 0).
  팔레트가 프레임별 상대 눈금이라 절대 온도는 센서로 보충.
- --noimg: 이미지 입력을 0으로 (센서 MLP만; 이미지 기여 분리용 대조군)
- 클래스 가중: 1/n^cw_pow (기본 1.0 → 짧은 4·5·6단계 강조), 긴 실험 500행 상한 비중.
출력: ~/jaehyeok/cache/img/cnn_{tag}.npz (out{k}_eval), cnn_{tag}_out{k}.pt
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import imglib as L
from imglib import cv

torch.set_num_threads(4)
ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--sens', type=int, default=0)
ap.add_argument('--noimg', action='store_true')
ap.add_argument('--cw_pow', type=float, default=1.0)
ap.add_argument('--epochs', type=int, default=8)
ap.add_argument('--epoch_size', type=int, default=24000)
ap.add_argument('--bs', type=int, default=128)
ap.add_argument('--lr', type=float, default=5e-4)
ap.add_argument('--full', action='store_true')
args = ap.parse_args()
dev = 'cuda'
pool, evalp = L.load_all()
N = len(pool)
y = torch.tensor(pool.label.to_numpy() - 1, device=dev)
is_eval = np.zeros(N, bool); is_eval[pool.index.get_indexer(evalp.index)] = True
cp = L.context_index(pool, np.ones(N, bool), lags=(5, 10, 15))
ce = L.context_index(pool, is_eval, lags=())
T = lambda a: torch.as_tensor(a, device=dev)
P_prev = torch.stack([T(cp[f'lag{w}']) for w in (5, 10, 15)]); E_prev = T(np.where(ce['prev'] >= 0, ce['prev'], 0))
X = torch.from_numpy(np.load(L.ICACHE / 'pool_rgb128.npy')).to(dev)
MEAN = torch.tensor([0.485, 0.456, 0.406] * 2, device=dev).view(1, 6, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225] * 2, device=dev).view(1, 6, 1, 1)
tt = T(pool.relative_time_s.to_numpy(np.float32))
RAW = T(np.nan_to_num(pool[['meas_t1', 'meas_t2', 'meas_v', 'meas_p']].to_numpy(np.float32), nan=0.0))


def sens_feat(idx, prev):
    a, b = RAW[idx], RAW[prev]
    lt = torch.log1p(a[:, :2].clamp(min=0))
    d = a - b
    return torch.cat([lt, a[:, 2:], d[:, :2] / 10, d[:, 2:], ((tt[idx] - tt[prev]) / 10)[:, None]], 1)


class Net(nn.Module):
    def __init__(self, sens):
        super().__init__()
        self.backbone = timm.create_model('resnet18.tv_in1k', pretrained=True, num_classes=0, in_chans=9)
        self.sens = sens
        if sens:
            self.mlp = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
            self.register_buffer('mu', torch.zeros(9)); self.register_buffer('sd', torch.ones(9))
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(512 + (64 if sens else 0), 6)

    def forward(self, x, s=None):
        h = self.drop(self.backbone(x))
        if self.sens:
            h = torch.cat([h, self.mlp((s - self.mu) / self.sd)], 1)
        return self.head(h)


def batch(idx, ctx, train):
    if ctx == 'pool':
        prev = P_prev[torch.randint(0, 3, idx.shape, device=dev) if train else torch.zeros_like(idx), idx]
    else:
        prev = E_prev[idx]
    x = torch.cat([X[idx], X[prev]], -1).permute(0, 3, 1, 2).float().div_(255)
    if train:
        B = len(idx)
        s = 1 + (torch.rand(B, device=dev) - .5) * .3
        flip = (torch.rand(B, device=dev) < .5).float() * 2 - 1
        th = torch.zeros(B, 2, 3, device=dev)
        th[:, 0, 0] = s * flip; th[:, 1, 1] = s * (1 + (torch.rand(B, device=dev) - .5) * .1)
        th[:, :, 2] = (torch.rand(B, 2, device=dev) - .5) * .2
        x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode='bilinear', padding_mode='border', align_corners=False)
    diff = (x[:, :3] - x[:, 3:]) * 2                      # 부호 있는 차이 채널
    x = torch.cat([(x - MEAN) / STD, diff], 1)
    if args.noimg:
        x = torch.zeros_like(x)
    return x.contiguous(memory_format=torch.channels_last), (sens_feat(idx, prev) if args.sens else None)


folds = cv.make_folds(pool, n_repeats=1)
fold_of = np.full(N, -1)
for f in folds:
    fold_of[pool.experiment_id.isin(f['valid_exps']).to_numpy()] = f['fold']
n_exp = pool.groupby('experiment_id')['label'].transform('size').to_numpy()
base_w = np.minimum(1.0, cv.CAP / n_exp)


def train_model(train_mask, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(args.sens).to(dev).to(memory_format=torch.channels_last)
    idx_all = np.where(train_mask)[0]
    if args.sens:                                          # 표준화 통계는 학습 fold에서만
        sf = sens_feat(T(idx_all), P_prev[0, T(idx_all)])
        m.mu.copy_(sf.mean(0)); m.sd.copy_(sf.std(0) + 1e-6)
    yy = pool.label.to_numpy()[idx_all]
    cw = (1.0 / np.bincount(yy, minlength=7)[1:].clip(1)) ** args.cw_pow
    w = torch.tensor(base_w[idx_all] * cw[yy - 1], device=dev, dtype=torch.float)
    idx_all = T(idx_all)
    steps = args.epochs * (args.epoch_size // args.bs)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    m.train()
    g = torch.Generator(device=dev); g.manual_seed(seed)
    for it in range(steps):
        idx = idx_all[torch.multinomial(w, args.bs, replacement=True, generator=g)]
        xb, sb = batch(idx, 'pool', True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(m(xb, sb), y[idx], label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return m.eval()


@torch.no_grad()
def predict(m, rows):
    out = np.full((N, 6), np.nan, dtype=np.float32)
    rows_t = T(rows)
    for i in range(0, len(rows), 512):
        xb, sb = batch(rows_t[i:i + 512], 'eval', False)
        out[rows[i:i + 512]] = F.softmax(m(xb, sb).float(), 1).cpu().numpy()
    return out


if args.full:
    m = train_model(np.ones(N, bool), 999)
    torch.save({n: v.half() if v.is_floating_point() else v for n, v in m.state_dict().items()}, L.ICACHE / f'cnn_{args.tag}_full.pt')
    print('full done', flush=True); raise SystemExit
res = {}
for k in range(5):
    t0 = time.time()
    hold = fold_of == k
    m = train_model(~hold, 100 + k)
    rows = np.where(hold & is_eval)[0]
    pe = predict(m, rows)
    res[f'out{k}_eval'] = pe
    f1 = cv.macro_f1(pool.label.to_numpy()[rows], pe[rows].argmax(1) + 1)
    print(f'out{k}: held-out evalp rows {len(rows)} macroF1 {f1:.4f} ({time.time() - t0:.0f}s)', flush=True)
    torch.save({n: v.half() if v.is_floating_point() else v for n, v in m.state_dict().items()}, L.ICACHE / f'cnn_{args.tag}_out{k}.pt')
    np.savez(L.ICACHE / f'cnn_{args.tag}.npz', **res)
print('done', flush=True)
