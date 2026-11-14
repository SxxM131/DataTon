"""pairdiff CNN(현재 + 직전 프레임 + 차이, finetune2.py --tag pairdiff) 최종 추론 원형 + 결정성 점검 (Test는 추론만).

직전 프레임 = 입력 CSV에 있는 같은 실험의 행 중 시간상 바로 앞 관측 (실험 첫 행은 자기 자신 → 차이 0). 과거만 사용.
1장씩(배치 1) + TF32 off → 배치 구성·행 순서·다른 실험 유무·미래 행 유무와 무관하게 비트 단위 동일.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm

from infer_bench import decode

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=0, in_chans=9)   # 구조만 (다운로드 없음)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(512, 6)

    def forward(self, x):
        return self.head(self.drop(self.backbone(x)))


def load_model(path, dev):
    m = Net()
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in torch.load(path, map_location='cpu').items()})
    return m.to(dev).eval()


def prev_index(frame):
    t = pd.to_numeric(frame['relative_time_s']).to_numpy(float)
    sid = frame['sample_id'].to_numpy().astype(str)
    prev = np.arange(len(frame))
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.lexsort((sid[idx], t[idx]))]
        prev[idx[1:]] = idx[:-1]
    return prev


@torch.no_grad()
def cnn_probs(frame, image_root, model, dev, bs=1):
    frame = frame.reset_index(drop=True)
    imgs = torch.from_numpy(np.stack([decode(Path(image_root) / p) for p in frame['image_path']]))
    prev = prev_index(frame)
    mean = torch.tensor([0.485, 0.456, 0.406] * 2, device=dev).view(1, 6, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * 2, device=dev).view(1, 6, 1, 1)
    out = np.zeros((len(frame), 6), np.float32)
    for i in range(0, len(frame), bs):
        j = np.arange(i, min(i + bs, len(frame)))
        x = torch.cat([imgs[j], imgs[prev[j]]], -1).to(dev).permute(0, 3, 1, 2).float().div(255)
        x = torch.cat([(x - mean) / std, (x[:, :3] - x[:, 3:]) * 2], 1)
        out[j] = torch.softmax(model(x).float(), 1).cpu().numpy()
    return out


if __name__ == '__main__':
    root = Path.home() / 'data/battery_data_light/test'
    te = pd.read_csv(root / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    m = load_model(Path.home() / 'jaehyeok/cache/img/cnn_pairdiff_full.pt', 'cuda')
    t0 = time.time(); p = cnn_probs(te, root, m, 'cuda'); print(f'{len(te)}장 디코딩+추론(1장씩) {time.time() - t0:.1f}s', flush=True)
    sh = te.sample(frac=1, random_state=0)
    p3 = np.empty_like(p); p3[sh.index.to_numpy()] = cnn_probs(sh, root, m, 'cuda')
    one = te[te.experiment_id == te.experiment_id.iloc[0]]
    past = te.sort_values('relative_time_s').groupby('experiment_id').head(60)
    for name, a, b in [('같은 입력 재실행', p, cnn_probs(te, root, m, 'cuda')), ('행 순서 섞기', p, p3), ('실험 1개만', p[one.index], cnn_probs(one, root, m, 'cuda')),
                       ('미래 행 제거', p[past.index], cnn_probs(past, root, m, 'cuda'))]:
        print(f'   {name}: max|Δp| {np.abs(a - b).max():.2e}, 비트 단위 동일 {np.array_equal(a, b)}', flush=True)
    print('   예측 단계 분포(argmax):', np.bincount(p.argmax(1) + 1, minlength=7)[1:].tolist())
