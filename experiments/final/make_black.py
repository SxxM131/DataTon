"""운영진 이미지 점검 재현용: Test 이미지와 같은 크기의 검은 이미지를 ~/jaehyeok/cache/final/black_test/ 아래에 만든다 (~/data는 읽기만)."""
from pathlib import Path
import pandas as pd
from PIL import Image
D = Path.home() / 'data/battery_data_light/test'
OUT = Path.home() / 'jaehyeok/cache/final/black_test'
te = pd.read_csv(D / 'test.csv', dtype=str)
for p in te.image_path:
    with Image.open(D / p) as im:
        size, mode = im.size, im.mode
    q = OUT / p
    q.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode if mode in ('RGB', 'RGBA', 'L', 'P') else 'RGB', size).save(q)
print('검은 이미지', len(te), '장 →', OUT, '(mode 예:', mode, ')')
