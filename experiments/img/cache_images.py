"""이미지 1회 디코딩 → 고정 크기 배열 캐시 (레포 밖 ~/jaehyeok/cache/img/).

저장물 (split = pool | test):
  {split}_rgb{S}.npy : (N, S, S, 3) uint8, 원본 RGB를 S×S로 bilinear 리사이즈 (종횡비 무시; 실험 내 프레이밍은 일정)
  {split}_pt{S}.npy  : (N, S, S) uint8, 팔레트 역변환 의사온도(0~255). 원본 해상도에서 계산 후 리사이즈
  {split}_ids.npy    : sample_id 순서
의사온도 정의는 exp01.color_stats와 동일 (고정 상수, 데이터로 학습하지 않음).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv  # noqa: E402

S = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DATA = Path.home() / 'data/battery_data_light'
OUT = Path.home() / 'jaehyeok/cache/img'


def pseudo_temp(rgb: Image.Image) -> np.ndarray:
    hsv = np.asarray(rgb.convert('HSV'), dtype=np.float32)
    h = hsv[..., 0] * (360.0 / 256.0)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    white = (s < 0.16) & (v > 0.78)
    pt = np.clip((300.0 - h) / 300.0, 0.0, 1.0)
    pt = np.where(h > 330.0, 1.0, pt)
    pt = np.where(white, 1.0, pt)
    return pt.astype(np.float32)


def one(path):
    with Image.open(path) as img:
        rgb = img.convert('RGB')
    pt = Image.fromarray(pseudo_temp(rgb), mode='F').resize((S, S), Image.BILINEAR)
    return (np.asarray(rgb.resize((S, S), Image.BILINEAR), dtype=np.uint8),
            np.clip(np.asarray(pt) * 255.0 + 0.5, 0, 255).astype(np.uint8))


def run(name, ids, files):
    out = Parallel(n_jobs=4, batch_size=256)(delayed(one)(p) for p in files)
    np.save(OUT / f'{name}_rgb{S}.npy', np.stack([o[0] for o in out]))
    np.save(OUT / f'{name}_pt{S}.npy', np.stack([o[1] for o in out]))
    np.save(OUT / f'{name}_ids.npy', np.asarray(ids, dtype=str))
    print(name, len(files), flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    pool = cv.load_pool(DATA)
    run('pool', pool.sample_id.values, pool.image_file.values)
    if '--test' in sys.argv:   # 추론 시간 측정/최종 추론 리허설용 (학습·통계 산출에 쓰지 않음)
        te = pd.read_csv(DATA / 'test/test.csv', dtype={'sample_id': 'string'})
        run('test', te.sample_id.values, (str(DATA / 'test') + '/' + te.image_path.astype(str)).values)
