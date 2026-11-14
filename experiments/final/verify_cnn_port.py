"""CNN 이식 확인: 루트 model.cnn_probs == experiments/img/infer_pair.cnn_probs (같은 가중치, Test 전체, GPU)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments/img')]
import model as M  # noqa: E402
import infer_pair  # noqa: E402
D = Path.home() / 'data/battery_data_light/test'
te = pd.read_csv(D / 'test.csv', dtype={'sample_id': 'string', 'experiment_id': 'string'})
w = ROOT / 'weights/cnn_pairdiff.pt'
a = M.cnn_probs(te, M.decode_images([D / p for p in te.image_path]), M.load_cnn(w, 'cuda'), 'cuda')
b = infer_pair.cnn_probs(te, D, infer_pair.load_model(w, 'cuda'), 'cuda')
print('model.cnn_probs vs infer_pair.cnn_probs 비트 동일:', np.array_equal(a, b), '최대차', np.abs(a - b).max())
p = pd.read_pickle(Path.home() / 'jaehyeok/cache/final/run1_proba.pkl')
print('test.py 저장 p_cnn 과 동일:', np.array_equal(p['p_cnn'], a))
