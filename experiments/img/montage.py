"""단계 경계 전후 프레임 몽타주 (육안 확인용). 출력은 스크래치 폴더."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'common')]
import cv

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
pool = cv.load_pool(Path.home() / 'data/battery_data_light')
et = cv.exp_table(pool)
OFFS = [-60, -20, -5, 0, 5, 20, 60]
T = 110

def montage(eid):
    d = pool[pool.experiment_id == eid].sort_values('relative_time_s').reset_index(drop=True)
    lab = d.label.to_numpy(); t = d.relative_time_s.to_numpy()
    bounds = [0] + [i for i in range(1, len(d)) if lab[i] != lab[i - 1]]
    W = len(OFFS) * (T + 4); H = len(bounds) * (T + 16)
    can = Image.new('RGB', (W, H), 'black'); dr = ImageDraw.Draw(can)
    for r, b in enumerate(bounds):
        for c, o in enumerate(OFFS):
            j = int(np.argmin(np.abs(t - (t[b] + o))))
            if abs(t[j] - (t[b] + o)) > 6: continue
            im = Image.open(d.image_file[j]).convert('RGB').resize((T, T), Image.NEAREST)
            can.paste(im, (c * (T + 4), r * (T + 16) + 14))
            dr.text((c * (T + 4) + 2, r * (T + 16) + 1), f'L{lab[j]} t={int(t[j])}', fill='white')
    can.save(OUT / f'{eid[:8]}_b{et.bat_class[eid]}_cc{et.bat_cc[eid]:g}_m{et.tr_method[eid]}.png')

rng = np.random.default_rng(0)
for (b, cc), g in et.groupby(['bat_class', 'bat_cc']):
    for eid in rng.choice(g.index, min(2, len(g)), replace=False):
        montage(eid)
print(sorted(p.name for p in OUT.glob('*.png')))
