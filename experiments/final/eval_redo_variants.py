"""공식 분할(Train 78 → Validation 19, Test식 행)에서 2·3단계 재판정에 이미지를 섞을지 비교. CNN 시드 3개."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import model as M  # noqa: E402
import train as T  # noqa: E402

pool = T.load_pool(Path.home() / 'data' / 'battery_data_light')
color = pd.concat([pd.read_pickle(Path.home() / 'jaehyeok' / 'cache' / f'{s}_color.pkl') for s in ('train', 'validation')])
y = pool.label.to_numpy(int)
Xp = M.build_features(pool, color); columns = list(Xp.columns); c23 = columns + M.Q_COLS
tr = (pool.origin == 'train').to_numpy()
m = M.build_model().fit(Xp[tr], y[tr])
er = M.pressure_ranges(pool[tr]); tab = M.range_table(er)
m23 = M.build_model().fit(pd.concat([Xp[tr], M.pressure_features(pool[tr], tab, own_range=M.loo_ranges(er))], axis=1)[c23], y[tr])
ev = T.to_test_like(pool[pool.origin == 'validation'])
Xe = M.build_features(ev, color)[columns]
pA = m.predict_proba(Xe); pB = m23.predict_proba(pd.concat([Xe, M.pressure_features(ev, tab)], axis=1)[c23])
ye = ev.label.to_numpy(int)
f1 = lambda p: round(T.macro_f1(ye, p), 4)  # noqa: E731
print('테이블 기본', f1(M.to_label(pA)), '| 테이블 + 재판정', f1(M.redecide_23(M.to_label(pA), pB)), flush=True)
images = M.decode_images(pool['image_file'].tolist())
pos = pool.index.get_indexer(ev.index)
for seed in (100, 101, 102):
    cnn = T.train_cnn(pool, images, tr, seed=seed)
    M.set_torch_deterministic()
    pC = M.cnn_probs(ev, images[pos], cnn, 'cuda')
    row = {'CNN 단독': f1(M.to_label(pC))}
    for w in (0.1, 0.2):
        fa = M.to_label(M.fuse(pA, pC, w))
        row[f'w={w} 융합'] = f1(fa)
        row[f'w={w} 융합→재판정(B+C)'] = f1(M.redecide_23(fa, M.fuse(pB, pC, w)))
        row[f'w={w} 융합→재판정(B 단독)'] = f1(M.redecide_23(fa, pB))
        x = M.redecide_23(M.to_label(pA), pB)
        row[f'w={w} 재판정 후 4~6만 융합'] = f1(np.where((x >= 4) | (fa >= 4), fa, x))
    print(f'seed {seed}:', row, flush=True)
    del cnn
