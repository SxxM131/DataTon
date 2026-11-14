"""최종 추론 리허설 (Test는 추론만): PNG 디코딩 → 9ch 입력(현재 + 30초 이하 과거 프레임 + 첫 프레임) → CNN 확률.

점검: (1) 3,451장 처리 시간  (2) 배치 크기·행 순서·실험 분할·미래 행 제거에 결과가 의존하는지 (비트 단위 비교)
test.py에 넣을 함수의 원형: cnn_probs(frame, image_root, weights) → (len(frame), 6) — frame 행만 보고, 실험별 과거 프레임만 사용.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image

torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
S = 128


def load_model(path, dev):
    sd = {k: v.float() for k, v in torch.load(path, map_location='cpu').items()}
    m = timm.create_model('resnet18', pretrained=False, num_classes=6, in_chans=sd['conv1.weight'].shape[1])   # 구조만 생성 (다운로드 없음). 3ch=현재 프레임, 9ch=시간 문맥
    m.load_state_dict(sd)
    return m.to(dev).eval()


def decode(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((S, S), Image.BILINEAR), dtype=np.uint8)


def context(frame):
    """행 위치 기준 (lag, first): 같은 실험에서 t <= t_i - 30인 마지막 관측(없으면 첫 관측), 첫 관측. 입력 행만 사용."""
    t = pd.to_numeric(frame['relative_time_s']).to_numpy(float)
    n = len(frame)
    lag, first = np.zeros(n, int), np.zeros(n, int)
    for _, idx in frame.groupby('experiment_id', sort=False).indices.items():
        idx = idx[np.lexsort((frame['sample_id'].to_numpy().astype(str)[idx], t[idx]))]   # 시간순 (동률은 sample_id로 고정)
        tt = t[idx]
        j = np.searchsorted(tt, tt - 30.0, side='right') - 1
        lag[idx] = idx[np.maximum(j, 0)]; first[idx] = idx[0]
    return lag, first


@torch.no_grad()
def cnn_probs(frame, image_root, model, dev, bs=256):
    frame = frame.reset_index(drop=True)
    imgs = torch.from_numpy(np.stack([decode(Path(image_root) / p) for p in frame['image_path']]))
    lag, first = context(frame)
    C = model.conv1.weight.shape[1]
    mean = torch.tensor([0.485, 0.456, 0.406] * (C // 3), device=dev).view(1, C, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225] * (C // 3), device=dev).view(1, C, 1, 1)
    out = np.zeros((len(frame), 6), np.float32)
    for i in range(0, len(frame), bs):
        j = np.arange(i, min(i + bs, len(frame)))
        x = imgs[j] if C == 3 else torch.cat([imgs[j], imgs[lag[j]], imgs[first[j]]], -1)
        x = x.to(dev).permute(0, 3, 1, 2).float().div(255)
        out[j] = torch.softmax(model((x - mean) / std).float(), 1).cpu().numpy()
    return out


if __name__ == '__main__':
    root = Path.home() / 'data/battery_data_light/test'
    te = pd.read_csv(root / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
    w = Path.home() / 'jaehyeok/cache/img' / next((a for a in sys.argv[1:] if a.endswith('.pt')), 'cnn_temporal_out0.pt')
    sh = te.sample(frac=1, random_state=0)
    one = te[te.experiment_id == te.experiment_id.iloc[0]]
    past = te.sort_values('relative_time_s').groupby('experiment_id').head(60)

    def check(dev, bs_main, bs_alt, label):
        m = load_model(w, dev)
        t0 = time.time(); p = cnn_probs(te, root, m, dev, bs=bs_main); t1 = time.time() - t0
        print(f'[{label}] {len(te)}장 디코딩+추론 {t1:.1f}s (배치 {bs_main})', flush=True)
        p2 = cnn_probs(te, root, m, dev, bs=bs_alt[0])
        p3 = np.empty_like(p); p3[sh.index.to_numpy()] = cnn_probs(sh, root, m, dev, bs=bs_alt[1])
        p4 = cnn_probs(one, root, m, dev, bs=bs_alt[2])
        p5 = cnn_probs(past, root, m, dev, bs=bs_alt[3])
        for name, a, b in [(f'배치 {bs_main} vs {bs_alt[0]}', p, p2), ('행 순서 섞기', p, p3), ('실험 1개만', p[one.index], p4), ('미래 행 제거', p[past.index], p5)]:
            print(f'   {name}: max|Δp| {np.abs(a - b).max():.2e}, 비트 단위 동일 {np.array_equal(a, b)}', flush=True)
        return p

    if 'tf32on' in sys.argv:
        check('cuda', 256, (7, 100, 33, 64), 'GPU TF32 on(기본)')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    check('cuda', 256, (7, 100, 33, 64), 'GPU TF32 off, 가변 배치')
    check('cuda', 1, (1, 1, 1, 1), 'GPU TF32 off, 1장씩')
    if 'cpu' in sys.argv:
        check('cpu', 64, (7, 100, 33, 64), 'CPU 4스레드, 가변 배치')
